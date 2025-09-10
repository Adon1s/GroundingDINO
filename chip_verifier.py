# Zillow House Investment Analyzer - LLM Vision Analysis
# Analyzes Zillow listings using LLM with structured JSON output

import fire
import base64
import requests
import json
from pathlib import Path
from typing import Dict, Optional, List
import time
import warnings
import sys
import os

# Force UTF-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

# Configuration
LM_STUDIO_URL = "http://192.168.86.143:1234"  # Adjust to your LM Studio URL
LISTINGS_JSON = Path("zillow_listings.json")
IMAGES_DIR = Path("zillow_images")
ANALYSIS_OUTPUT = Path("zillow_house_analysis_results.json")

# Debug mode
DEBUG_MODE = True

# JSON Schema for House Analysis
HOUSE_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "zillow_house_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "investment_score": {
                    "type": "integer",
                    "description": "1-10 overall investment potential"
                },
                "property_condition_score": {
                    "type": "integer",
                    "description": "1-10 visible property condition rating"
                },
                "curb_appeal_score": {
                    "type": "integer",
                    "description": "1-10 curb appeal and attractiveness"
                },
                "rental_potential_score": {
                    "type": "integer",
                    "description": "1-10 potential as rental property"
                },
                "neighborhood_quality": {
                    "type": "string",
                    "enum": ["Excellent", "Good", "Average", "Below Average", "Poor"]
                },
                "property_type_classification": {
                    "type": "string",
                    "enum": ["Single Family", "Townhouse", "Condo", "Multi-Family", "Other"]
                },
                "renovation_needs": {
                    "type": "string",
                    "enum": ["None/Minimal", "Light", "Moderate", "Heavy", "Complete"]
                },
                "primary_strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 main property strengths"
                },
                "primary_concerns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 main concerns or issues"
                },
                "estimated_monthly_rent": {
                    "type": "object",
                    "properties": {
                        "low": {"type": "number"},
                        "high": {"type": "number"}
                    },
                    "required": ["low", "high"]
                },
                "price_assessment": {
                    "type": "string",
                    "enum": ["Underpriced", "Fair", "Slightly High", "Overpriced"]
                },
                "investment_verdict": {
                    "type": "string",
                    "enum": ["STRONG BUY", "CONSIDER", "PASS", "AVOID"]
                },
                "verdict_reasoning": {
                    "type": "string",
                    "description": "2-3 sentence investment recommendation"
                },
                "ideal_buyer_profile": {
                    "type": "string",
                    "description": "Who would be best suited for this property"
                },
                "quick_sale_likelihood": {
                    "type": "string",
                    "enum": ["Very High", "High", "Moderate", "Low", "Very Low"]
                },
                "suggested_offer_percentage": {
                    "type": "integer",
                    "description": "Suggested offer as percentage of asking (e.g., 95 for 95%)"
                }
            },
            "required": [
                "investment_score",
                "property_condition_score",
                "curb_appeal_score",
                "rental_potential_score",
                "neighborhood_quality",
                "property_type_classification",
                "renovation_needs",
                "primary_strengths",
                "primary_concerns",
                "estimated_monthly_rent",
                "price_assessment",
                "investment_verdict",
                "verdict_reasoning",
                "ideal_buyer_profile",
                "quick_sale_likelihood",
                "suggested_offer_percentage"
            ]
        }
    }
}


def create_house_analysis_prompt(listing: Dict) -> str:
    """Generate the analysis prompt for a house listing"""
    # Calculate price per sqft if possible
    price_per_sqft = "N/A"
    if listing.get('sqft') and listing['sqft'] != 'N/A':
        # Extract number from sqft string like "2,859 sqft"
        try:
            sqft_num = int(listing['sqft'].replace(',', '').replace('sqft', '').strip())
            price_num = float(listing['price'].replace('$', '').replace(',', ''))
            price_per_sqft = f"${price_num / sqft_num:.2f}"
        except:
            pass

    return f"""You are an expert real estate investment analyst evaluating properties in Huntsville, Alabama.

PROPERTY TO ANALYZE:
- Address: {listing.get('address', 'N/A')}
- Asking Price: {listing.get('price', 'N/A')}
- Bedrooms: {listing.get('beds', 'N/A')}
- Bathrooms: {listing.get('baths', 'N/A')}
- Square Feet: {listing.get('sqft', 'N/A')}
- Price per Sqft: {price_per_sqft}
- Time on Market: {listing.get('time_posted', 'N/A')}
- Listing Agent: {listing.get('listing_agent', 'N/A')}

REQUIRED ANALYSIS:
1. investment_score (1-10): Overall investment potential
2. property_condition_score (1-10): Visible condition from photos
3. curb_appeal_score (1-10): Street appeal and attractiveness
4. rental_potential_score (1-10): Viability as rental property
5. neighborhood_quality: Based on visible surroundings
6. property_type_classification: Type of property
7. renovation_needs: Level of renovation required
8. primary_strengths: 3-5 main selling points
9. primary_concerns: 3-5 main issues or red flags
10. estimated_monthly_rent: {{low: X, high: Y}} for Huntsville market
11. price_assessment: Is it priced appropriately?
12. investment_verdict: "STRONG BUY", "CONSIDER", "PASS", or "AVOID"
13. verdict_reasoning: 2-3 sentences explaining recommendation
14. ideal_buyer_profile: Best suited buyer type
15. quick_sale_likelihood: How fast will it sell?
16. suggested_offer_percentage: What % of asking to offer (e.g., 95)

Consider Huntsville's tech growth, proximity to Research Park, and military presence.
Base rental estimates on: 0.8-1.2% of purchase price monthly for this market.

You MUST provide ALL fields based on the images and data provided."""


def parse_llm_response(response: str) -> Dict:
    """Parse the LLM response JSON string with validation"""
    try:
        parsed = json.loads(response)

        if DEBUG_MODE:
            # Check for missing fields
            expected_fields = [
                'investment_score', 'property_condition_score', 'curb_appeal_score',
                'rental_potential_score', 'neighborhood_quality', 'property_type_classification',
                'renovation_needs', 'primary_strengths', 'primary_concerns',
                'estimated_monthly_rent', 'price_assessment', 'investment_verdict',
                'verdict_reasoning', 'ideal_buyer_profile', 'quick_sale_likelihood',
                'suggested_offer_percentage'
            ]

            missing_fields = [f for f in expected_fields if f not in parsed]
            if missing_fields:
                print(f"⚠️  DEBUG - Missing fields: {missing_fields}")

        return parsed

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return {
            "error": "Invalid JSON",
            "raw_response": response[:500],
            "investment_verdict": "PASS",
            "verdict_reasoning": "Analysis failed"
        }


def analyze_house_remote(listing: Dict, image_paths: List[Path], model_name: Optional[str] = None) -> Dict:
    """Analyze a house via LM Studio REST API with structured output"""
    # Prepare images - use first 3 images max to save tokens
    image_data_list = []
    for img_path in image_paths[:3]:
        if img_path.exists():
            with open(img_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
                image_data_list.append(image_data)

    if not image_data_list:
        return {
            "error": "No images found",
            "investment_verdict": "PASS",
            "verdict_reasoning": "No images available for analysis"
        }

    prompt = create_house_analysis_prompt(listing)

    # Build message content with multiple images
    content = [{"type": "text", "text": prompt}]
    for img_data in image_data_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
        })

    messages = [
        {
            "role": "system",
            "content": "You are an expert real estate analyst specializing in investment properties. Provide thorough analysis based on visual inspection and market knowledge."
        },
        {
            "role": "user",
            "content": content
        }
    ]

    payload = {
        "model": model_name or "gemma-3-27b-it",
        "messages": messages,
        "response_format": HOUSE_ANALYSIS_SCHEMA,
        "temperature": 0.3,
        "max_tokens": 2000,
        "stream": False
    }

    try:
        if DEBUG_MODE:
            print(f"📡 Sending request to LM Studio...")
            print(f"   Using {len(image_data_list)} images")

        resp = requests.post(f"{LM_STUDIO_URL}/v1/chat/completions", json=payload, timeout=120)

        if DEBUG_MODE:
            print(f"📡 Response status: {resp.status_code}")

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {
            "error": f"Request failed: {e}",
            "investment_verdict": "PASS",
            "verdict_reasoning": "Analysis request failed"
        }

    if resp.status_code != 200:
        print(f"❌ HTTP error {resp.status_code}")
        return {
            "error": f"HTTP {resp.status_code}",
            "investment_verdict": "PASS",
            "verdict_reasoning": "Server error"
        }

    try:
        data = resp.json()
    except ValueError:
        return {
            "error": "Invalid JSON from server",
            "investment_verdict": "PASS",
            "verdict_reasoning": "Server response error"
        }

    if "choices" not in data or not data["choices"]:
        return {
            "error": "Missing choices in response",
            "investment_verdict": "PASS",
            "verdict_reasoning": "Incomplete server response"
        }

    content = data["choices"][0]["message"].get("content", "")

    if DEBUG_MODE and "usage" in data:
        usage = data["usage"]
        print(f"📊 Tokens - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
              f"Completion: {usage.get('completion_tokens', 'N/A')}")

    return parse_llm_response(content)


def get_listing_images(listing_address: str, listing_index: int) -> List[Path]:
    """Find all images for a listing"""
    # Clean address for folder name
    folder_name = listing_address.replace('/', '_').replace(',', '').replace('\\', '_').replace(':', '')

    # Try both naming conventions
    possible_dirs = [
        IMAGES_DIR / f"{listing_index}_{folder_name}",
        IMAGES_DIR / folder_name,
        IMAGES_DIR / f"listing_{listing_index}"
    ]

    for listing_dir in possible_dirs:
        if listing_dir.exists():
            # Find all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                image_files.extend(listing_dir.glob(ext))

            # Sort by name to get consistent order
            image_files.sort()
            return image_files

    return []


def analyze_single_house(listing: Dict, listing_index: int, model_name: Optional[str] = None) -> Dict:
    """Analyze a single house listing"""
    address = listing.get('address', 'Unknown')

    print(f"\n{'=' * 60}")
    print(f"Analyzing: {address}")
    print(
        f"Price: {listing.get('price', 'N/A')} | {listing.get('beds', 'N/A')} | {listing.get('baths', 'N/A')} | {listing.get('sqft', 'N/A')}")

    # Find images
    image_paths = get_listing_images(address, listing_index)

    if not image_paths:
        print(f"⚠️  No images found for {address}")
        return {
            'address': address,
            'listing_data': listing,
            'analysis': {
                'error': 'No images found',
                'investment_verdict': 'PASS',
                'verdict_reasoning': 'Cannot analyze without images'
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    print(f"Found {len(image_paths)} images")

    # Analyze with LLM
    analysis = analyze_house_remote(listing, image_paths, model_name)

    return {
        'address': address,
        'listing_data': listing,
        'images_analyzed': len(image_paths),
        'analysis': analysis,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }


def load_listings() -> List[Dict]:
    """Load Zillow listings from JSON file"""
    if not LISTINGS_JSON.exists():
        print(f"❌ Listings file not found: {LISTINGS_JSON}")
        return []

    with open(LISTINGS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both single listings and arrays
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'listings' in data:
        return data['listings']
    else:
        # Assume it's a single listing
        return [data]


def load_existing_results() -> Dict[str, Dict]:
    """Load existing analysis results"""
    if not ANALYSIS_OUTPUT.exists():
        return {}

    try:
        with open(ANALYSIS_OUTPUT, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {r['address']: r for r in data.get('results', [])}
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing results: {e}")
        return {}


def append_result_to_file(result: Dict, existing_results: Dict[str, Dict]):
    """Append a single result to the output file"""
    existing_results[result['address']] = result
    all_results = list(existing_results.values())

    # Calculate summary stats
    strong_buy_count = sum(1 for r in all_results
                           if r.get('analysis', {}).get('investment_verdict') == 'STRONG BUY')
    consider_count = sum(1 for r in all_results
                         if r.get('analysis', {}).get('investment_verdict') == 'CONSIDER')
    pass_count = sum(1 for r in all_results
                     if r.get('analysis', {}).get('investment_verdict') == 'PASS')
    avoid_count = sum(1 for r in all_results
                      if r.get('analysis', {}).get('investment_verdict') == 'AVOID')

    # Calculate averages
    investment_scores = [r.get('analysis', {}).get('investment_score', 0)
                         for r in all_results if 'analysis' in r and 'error' not in r['analysis']]
    avg_investment = sum(investment_scores) / len(investment_scores) if investment_scores else 0

    output_data = {
        'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_analyzed': len(all_results),
        'summary': {
            'strong_buy_count': strong_buy_count,
            'consider_count': consider_count,
            'pass_count': pass_count,
            'avoid_count': avoid_count,
            'average_investment_score': round(avg_investment, 2)
        },
        'results': all_results
    }

    with open(ANALYSIS_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved analysis for {result['address']}")


def print_house_analysis_summary(result: Dict):
    """Print formatted analysis summary"""
    analysis = result.get('analysis', {})

    print("\n📊 ANALYSIS RESULT:")
    print("-" * 60)

    if 'error' in analysis:
        print(f"❌ Error: {analysis.get('error')}")
        return

    # Scores
    print(f"Investment Score: {analysis.get('investment_score', 'N/A')}/10")
    print(f"Condition Score: {analysis.get('property_condition_score', 'N/A')}/10")
    print(f"Curb Appeal: {analysis.get('curb_appeal_score', 'N/A')}/10")
    print(f"Rental Potential: {analysis.get('rental_potential_score', 'N/A')}/10")

    # Key assessments
    print(f"\n📍 Verdict: {analysis.get('investment_verdict', 'N/A')}")
    print(f"   {analysis.get('verdict_reasoning', 'N/A')}")

    print(f"\n💰 Price Assessment: {analysis.get('price_assessment', 'N/A')}")
    print(f"   Suggested Offer: {analysis.get('suggested_offer_percentage', 'N/A')}% of asking")

    # Rental estimates
    if analysis.get('estimated_monthly_rent'):
        rent = analysis['estimated_monthly_rent']
        listing_price = result['listing_data'].get('price', '$0')
        try:
            price_num = float(listing_price.replace('$', '').replace(',', ''))
            rent_ratio_low = (rent['low'] * 12) / price_num * 100
            rent_ratio_high = (rent['high'] * 12) / price_num * 100
            print(f"\n🏠 Est. Monthly Rent: ${rent['low']:.0f} - ${rent['high']:.0f}")
            print(f"   Annual Return: {rent_ratio_low:.1f}% - {rent_ratio_high:.1f}%")
        except:
            print(f"\n🏠 Est. Monthly Rent: ${rent.get('low', 0):.0f} - ${rent.get('high', 0):.0f}")

    # Other details
    print(f"\n🏘️ Neighborhood: {analysis.get('neighborhood_quality', 'N/A')}")
    print(f"🔨 Renovation Needs: {analysis.get('renovation_needs', 'N/A')}")
    print(f"⚡ Quick Sale Likelihood: {analysis.get('quick_sale_likelihood', 'N/A')}")

    # Concerns
    if analysis.get('primary_concerns'):
        print(f"\n🚨 Main Concerns:")
        for concern in analysis['primary_concerns']:
            print(f"   - {concern}")

    # Strengths
    if DEBUG_MODE and analysis.get('primary_strengths'):
        print(f"\n✨ Main Strengths:")
        for strength in analysis['primary_strengths']:
            print(f"   - {strength}")


def analyze_houses(
        limit: Optional[int] = None,
        min_price: float = 0,
        max_price: float = 10000000,
        skip_analyzed: bool = True,
        model_name: Optional[str] = None
):
    """Analyze Zillow house listings

    Args:
        limit: Maximum number of houses to analyze
        min_price: Minimum price filter
        max_price: Maximum price filter
        skip_analyzed: Skip already analyzed houses
        model_name: Specific LM Studio model to use
    """
    print("\n=== Zillow House Investment Analyzer ===\n")

    # Load existing results
    existing_results = {}
    if skip_analyzed:
        existing_results = load_existing_results()
        print(f"Found {len(existing_results)} previously analyzed houses")

    # Load listings
    listings = load_listings()
    if not listings:
        print("No listings found!")
        return

    print(f"Loaded {len(listings)} total listings")

    # Filter listings
    filtered = []
    for i, listing in enumerate(listings):
        # Extract price for filtering
        try:
            price_str = listing.get('price', '$0')
            price_num = float(price_str.replace('$', '').replace(',', ''))
        except:
            price_num = 0

        address = listing.get('address', f'Unknown_{i}')

        if (not skip_analyzed or address not in existing_results) and \
                min_price <= price_num <= max_price:
            filtered.append((i + 1, listing))  # Store index with listing

    # Apply limit
    to_analyze = filtered[:limit] if limit else filtered

    print(f"Found {len(filtered)} houses matching criteria")
    print(f"Analyzing {len(to_analyze)} houses...\n")

    if not to_analyze:
        print("No houses to analyze!")
        return

    # Analyze each house
    analyzed_count = 0
    error_count = 0

    for i, (listing_index, listing) in enumerate(to_analyze, 1):
        print(f"\n[{i}/{len(to_analyze)}] Processing...")

        try:
            result = analyze_single_house(listing, listing_index, model_name)

            if 'error' in result.get('analysis', {}):
                error_count += 1
            else:
                analyzed_count += 1

            print_house_analysis_summary(result)

            if skip_analyzed:
                append_result_to_file(result, existing_results)

        except Exception as e:
            print(f"❌ Error analyzing house: {e}")
            error_count += 1

        # Delay between analyses
        if i < len(to_analyze):
            time.sleep(2)

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"✅ Successfully analyzed: {analyzed_count} houses")
    print(f"❌ Errors: {error_count} houses")
    if skip_analyzed:
        print(f"📊 Total in database: {len(existing_results)}")
    print(f"📁 Results saved to: {ANALYSIS_OUTPUT}")


def show_investment_opportunities(min_score: float = 7.0):
    """Show houses marked as investment opportunities"""
    results = load_existing_results()

    if not results:
        print("No houses have been analyzed yet.")
        return

    # Find opportunities
    opportunities = []

    for result in results.values():
        analysis = result.get('analysis', {})
        if 'error' in analysis:
            continue

        verdict = analysis.get('investment_verdict', '')
        score = analysis.get('investment_score', 0)

        if verdict in ['STRONG BUY', 'CONSIDER'] and score >= min_score:
            opportunities.append(result)

    # Sort by investment score
    opportunities.sort(key=lambda x: x['analysis']['investment_score'], reverse=True)

    print(f"\n=== Investment Opportunities (score >= {min_score}) ===")
    print(f"Found {len(opportunities)} opportunities\n")

    for i, opp in enumerate(opportunities, 1):
        listing = opp['listing_data']
        analysis = opp['analysis']

        print(f"{i}. {opp['address']}")
        print(
            f"   Price: {listing.get('price')} | {listing.get('beds')} | {listing.get('baths')} | {listing.get('sqft')}")
        print(f"   Investment Score: {analysis['investment_score']}/10")
        print(f"   Verdict: {analysis['investment_verdict']}")
        print(f"   Price Assessment: {analysis['price_assessment']}")

        # Calculate potential return
        if analysis.get('estimated_monthly_rent'):
            try:
                price_num = float(listing['price'].replace('$', '').replace(',', ''))
                monthly_rent_avg = (analysis['estimated_monthly_rent']['low'] +
                                    analysis['estimated_monthly_rent']['high']) / 2
                annual_return = (monthly_rent_avg * 12) / price_num * 100
                print(f"   Est. Annual Return: {annual_return:.1f}%")
            except:
                pass

        print(f"   {analysis['verdict_reasoning']}")
        print()


def export_best_deals(output_file: str = "best_zillow_deals.json", top_n: int = 10):
    """Export the best investment opportunities"""
    results = load_existing_results()

    if not results:
        print("No houses have been analyzed yet.")
        return

    # Score and rank all properties
    scored_properties = []

    for result in results.values():
        analysis = result.get('analysis', {})
        if 'error' in analysis:
            continue

        # Calculate composite score
        investment_score = analysis.get('investment_score', 0)
        rental_score = analysis.get('rental_potential_score', 0)
        condition_score = analysis.get('property_condition_score', 0)

        # Weight investment score more heavily
        composite_score = (investment_score * 0.5 +
                           rental_score * 0.3 +
                           condition_score * 0.2)

        # Get price metrics
        listing_price = result['listing_data'].get('price', '$0')
        try:
            price_num = float(listing_price.replace('$', '').replace(',', ''))

            # Calculate metrics
            suggested_offer_pct = analysis.get('suggested_offer_percentage', 100)
            suggested_offer = price_num * (suggested_offer_pct / 100)

            rent_est = analysis.get('estimated_monthly_rent', {})
            monthly_rent_avg = (rent_est.get('low', 0) + rent_est.get('high', 0)) / 2
            annual_return = (monthly_rent_avg * 12) / suggested_offer * 100 if suggested_offer > 0 else 0

            scored_properties.append({
                'address': result['address'],
                'listing_details': result['listing_data'],
                'asking_price': price_num,
                'suggested_offer': suggested_offer,
                'monthly_rent_estimate': {
                    'low': rent_est.get('low', 0),
                    'high': rent_est.get('high', 0),
                    'average': monthly_rent_avg
                },
                'estimated_annual_return': round(annual_return, 2),
                'scores': {
                    'composite': round(composite_score, 2),
                    'investment': investment_score,
                    'rental_potential': rental_score,
                    'condition': condition_score,
                    'curb_appeal': analysis.get('curb_appeal_score', 0)
                },
                'verdict': analysis.get('investment_verdict'),
                'price_assessment': analysis.get('price_assessment'),
                'renovation_needs': analysis.get('renovation_needs'),
                'ideal_buyer': analysis.get('ideal_buyer_profile'),
                'strengths': analysis.get('primary_strengths', []),
                'concerns': analysis.get('primary_concerns', []),
                'verdict_reasoning': analysis.get('verdict_reasoning')
            })
        except:
            continue

    # Sort by composite score
    scored_properties.sort(key=lambda x: x['scores']['composite'], reverse=True)

    # Get top N
    best_deals = scored_properties[:top_n]

    # Save to file
    output = {
        'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_properties': len(best_deals),
        'selection_criteria': 'Top properties by composite score (50% investment, 30% rental, 20% condition)',
        'market': 'Huntsville, AL',
        'best_deals': best_deals
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Exported top {len(best_deals)} investment opportunities to: {output_file}")

    # Show summary
    if best_deals:
        print("\n🏆 Top 5 Investment Opportunities:")
        for i, deal in enumerate(best_deals[:5], 1):
            print(f"\n{i}. {deal['address']}")
            print(f"   Asking: ${deal['asking_price']:,.0f} → Offer: ${deal['suggested_offer']:,.0f}")
            print(f"   Est. Return: {deal['estimated_annual_return']}% annually")
            print(f"   Composite Score: {deal['scores']['composite']}/10")
            print(f"   {deal['verdict']}: {deal['verdict_reasoning']}")


def analyze_specific_address(address: str, model_name: Optional[str] = None):
    """Analyze a specific house by address"""
    listings = load_listings()

    # Find the listing
    target_listing = None
    listing_index = None

    for i, listing in enumerate(listings):
        if address.lower() in listing.get('address', '').lower():
            target_listing = listing
            listing_index = i + 1
            break

    if not target_listing:
        print(f"❌ No listing found matching address: {address}")
        return

    # Check if already analyzed
    existing_results = load_existing_results()
    if target_listing['address'] in existing_results:
        print(f"ℹ️  This property has already been analyzed:")
        print_house_analysis_summary(existing_results[target_listing['address']])
        return

    # Analyze
    try:
        result = analyze_single_house(target_listing, listing_index, model_name)
        print_house_analysis_summary(result)
        append_result_to_file(result, existing_results)
    except Exception as e:
        print(f"❌ Error analyzing house: {e}")


def main():
    """Main entry point with Fire CLI"""
    fire.Fire({
        'analyze': analyze_houses,
        'analyze_address': analyze_specific_address,
        'opportunities': show_investment_opportunities,
        'export': export_best_deals
    })


if __name__ == "__main__":
    main()