#! /venv/bin/python


from DFTDataFrame.Tools import crawl, create_frame, get_element_counts
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Sample script with flags.")

# Add flags/arguments
parser.add_argument("-r", "--root", type=str, help="Root of calculations", required=True)
parser.add_argument("-o", "--output", type=str, help="Destination for Excel")
parser.add_argument("-q", "--query", type=str, help="Input for DataFrame.query()")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

# Parse arguments
args = parser.parse_args()

# Use the arguments
if args.verbose:
    print("[DEBUG] Parsed arguments:", args)

if args.root:
    print(f"Reading {args.root}.")
    paths = crawl(args.root)
    Frame = create_frame(root=args.root, calc_file='final.traj',  verbose=True)
    Frame = get_element_counts(Frame)


if args.query:
    print(f"Querying {args.query}.")
    Frame = Frame.query(query)

if args.output:
    Frame.to_excel(excel_writer=args.output+'.xlsx')


