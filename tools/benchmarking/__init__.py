"""Benchmark harness: frozen datasets, human reference truth, and evaluation.

Read-only with respect to production. Nothing in this package may import
Prisma, the analysis queue, or the scraper, or write outside a benchmark run
directory. See docs/ handoffs and the plan for the session breakdown.
"""
