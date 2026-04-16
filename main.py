"""
pySimpleDB — Query Optimization & Indexing Benchmark

Usage:
    python main.py --query Q1 --mode baseline
    python main.py --query Q1 --mode opt
    python main.py --query Q1 --mode index
    python main.py --query Q1 --mode full

    python main.py                          # runs all queries in baseline mode
    python main.py --help                   # show all options

Modes:
    baseline — No optimization, no indexes (reference)
    opt      — Join reordering + selection pushdown, no indexes
    index    — B-tree indexes, original join order
    full     — Both optimization and indexes
"""

from benchmark import main

if __name__ == "__main__":
    main()