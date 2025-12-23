# test_scene_classifier_stub.py
import asyncio
from pathlib import Path

from scene_classifier_orchestrator import (
    SceneClassifierOrchestrator,
    SceneClassifierRunOptions,
)
from tools.scene_classifier_orchestrator import create_orchestrator_from_config
from tools import pipeline_config as cfg

class StubVLMClient:
    async def analyze_image(self, image_path, system_prompt, user_prompt, **model_config):
        # Return different JSON depending on which pass we think is being run
        if "scene type" in system_prompt.lower():
            return """{"scene": "kitchen", "confidence": 0.92, "reasoning": "Looks like a kitchen"}"""
        if "overall impression" in system_prompt.lower():
            return """{
                "overall_impression":"Bright updated kitchen.", 
                "image_summary":"White cabinets, stainless appliances.",
                "notable_features":["stainless appliances","island"]
            }"""
        if "property inspector" in system_prompt.lower():
            return """{
              "detected_issues":[
                {"description":"Minor scuffs on cabinets","severity":"minor_repair","location":"lower cabinets","catalog_ids":["CABMINOR"],"confidence":0.7}
              ],
              "catalog_flags": {
                "CABMINOR":{"present":"yes","severity":"minor_repair","evidence":"visible scuffs"}
              }
            }"""
        if "quality control reviewer" in system_prompt.lower():
            return """{
              "verified":[{"issue_index":0,"reason":"visible in lower cabinets"}],
              "rejected":[],
              "notes":"Issues look reasonable"
            }"""
        if "extracting search keywords" in system_prompt.lower():
            return """{
              "keywords":["cabinet","stainless fridge","island"],
              "categories":{
                 "structural":["wall"],
                 "fixtures":["cabinet","fridge"],
                 "condition":["scuff"],
                 "style":["modern"]
              }
            }"""
        # Fallback JSON
        return "{}"

    async def analyze_text(self, system_prompt, user_prompt, **model_config):
        return """{
          "property_summary":"Updated kitchen with minor cosmetic wear.",
          "investment_considerations":["Mostly cosmetic work","No major defects visible"],
          "estimated_condition":"good",
          "confidence":0.85
        }"""


async def main():
    import pipeline_config as cfg
    from tools.vlm_client import get_model_configs_from_pipeline_config

    qwen_config, gpt5_config = get_model_configs_from_pipeline_config(cfg)

    orchestrator = SceneClassifierOrchestrator(
        qwen_config=qwen_config,
        gpt5_config=gpt5_config,
        vlm_client=StubVLMClient(),
        issue_catalog=None,
        max_keywords=10,
    )

    img = Path("/absolute/path/to/a/test.jpg")  # point at any real image; stub ignores contents

    options = SceneClassifierRunOptions(
        premium=True,  # exercise premium routing
    )

    property_result = await orchestrator.analyze_property(
        property_key="test_property",
        image_paths=[img],
        options=options,
    )

    print("Property summary:", property_result.property_summary)
    print("Total issues:", property_result.total_issues)
    for r in property_result.image_results:
        print("Image:", r.image_path)
        print("  scene:", r.scene)
        print("  impression:", r.overall_impression)
        print("  verified issues:", len(r.verified_issues))
        print("  models used:", r.models_used)


if __name__ == "__main__":
    asyncio.run(main())
