import argparse
import concurrent.futures
import json
import logging
import time
from pathlib import Path

import pandas as pd

from vulnerability_networks.algorithms.functionality_based import global_efficiency, number_independent_paths
from vulnerability_networks.algorithms.rank_links import process_network
from vulnerability_networks.config import PATH_INTERIM_DATA, PATH_RAW_DATA
from vulnerability_networks.logging import logger


# EXAMPLE SEQUENTIAL
def run_sequential():
    """run sequential example"""
    with open(PATH_RAW_DATA / "network_catalog.json", "r") as f:
        catalog = json.load(f)

    df = pd.DataFrame(catalog)
    df["path"] = df["path"].apply(lambda network_path: PATH_RAW_DATA / network_path / "graph.gml")

    # df_example = df.sort_values(by="nodes")[:10]
    df_example = df.query("network_id==3042")

    start_time = time.time()
    results_sequential = []
    for path in df_example["path"]:
        result = process_network(path, global_efficiency)
        results_sequential.append(result)
    print(f"[SEQUENTIAL] Total processing time: {time.time() - start_time:.2f} seconds")


# EXAMPLE PARALLEL SCENARIOS
def run_parallel():
    with open(PATH_RAW_DATA / "network_catalog.json", "r") as f:
        catalog = json.load(f)

    df = pd.DataFrame(catalog)
    df["path"] = df["path"].apply(lambda network_path: PATH_RAW_DATA / network_path / "graph.gml")

    df_example = df.query("size_category=='small'").sample(frac=1)

    start_time = time.time()
    results_parallel = []
    for path in df_example["path"]:
        result = process_network(path, global_efficiency, workers=16)
        results_parallel.append(result)
    print(f"[PARALLEL] Total processing time: {time.time() - start_time:.2f} seconds")


# EXAMPLE MULTIPARALLEL
def run_multiparallel():
    with open(PATH_RAW_DATA / "network_catalog.json", "r") as f:
        catalog = json.load(f)

    df = pd.DataFrame(catalog)
    df["path"] = df["path"].apply(lambda network_path: PATH_RAW_DATA / network_path / "graph.gml")

    df_example = df.sort_values(by="nodes")[:10]

    # MULTILEVEL PARALLELISM

    # Set up parameters
    inner_workers = 16  # Workers for scenario processing within each network
    outer_workers = 5  # Number of networks to process simultaneously

    # Start timing
    start_time = time.time()

    # Get list of network paths
    # network_paths = [path for path in df_example["path"]]
    results_multiparallel = []

    # Create outer parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=outer_workers) as executor:
        # Submit each network for processing
        future_to_path = {}
        for path in df_example["path"]:
            # Submit network processing job with inner parallelism
            future = executor.submit(
                process_network,
                path,
                global_efficiency,
                workers=inner_workers,  # Enable inner parallelism
                show_progress=False,  # Flag to avoid nested progress bars
            )
            future_to_path[future] = path

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results_multiparallel.append(result)
                print(f"Completed processing {path}")
            except Exception as e:
                print(f"Processing of {path} generated an exception: {e}")

    print(f"[MULTIPARALLEL] Total processing time: {time.time() - start_time:.2f} seconds")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Network Vulnerability Analysis")

    parser.add_argument(
        "--accessibility-index",
        type=str,
        choices=["global_efficiency"],
        default="global_efficiency",
        help="Accessibility index to use",
    )

    parser.add_argument("--max-links-ratio", type=float, default=0.4, help="Maximum ratio of links to disrupt")

    parser.add_argument("--max-scenarios", type=int, default=1_000_000, help="Maximum number of disruption scenarios")

    parser.add_argument("--inner-workers", type=int, default=16, help="Number of workers for scenario processing")

    parser.add_argument("--outer-workers", type=int, default=5, help="Number of networks to process simultaneously")

    return parser.parse_args()


def main():
    """Main function to run the analysis"""
    start_time = time.time()
    args = parse_arguments()

    # Set up paths
    # Load network catalog
    catalog_path = PATH_RAW_DATA / "network_catalog.json"
    logger.info(f"Loading network catalog from: {catalog_path}")
    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    # Prepare dataframe
    df = pd.DataFrame(catalog)
    df["path"] = df["path"].apply(lambda network_path: PATH_RAW_DATA / network_path / "graph.gml")

    df = df.query("size_category == 'small'").sample(frac=1)#.sort_values("nodes")[:10]

    # Set up accessibility index
    accessibility_indexes = {"global_efficiency": global_efficiency}
    index_fn = accessibility_indexes[args.accessibility_index]

    # Set up output directory
    folder_name = f"{args.accessibility_index}_ratiolinks{str(args.max_links_ratio).replace('.', '_')}_max_scenarios{args.max_scenarios}"
    path_to_save = PATH_INTERIM_DATA / folder_name
    path_to_save.mkdir(exist_ok=True, parents=True)

    logger.info(f"Starting analysis with {len(df)} networks")
    logger.info(f"Inner workers: {args.inner_workers}, Outer workers: {args.outer_workers}")
    logger.info(f"Results will be saved to: {path_to_save}")

    # Track completion metrics
    total_networks = len(df)
    completed_networks = 0
    failed_networks = 0

    try:
        # Create outer parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.outer_workers) as executor:
            # Submit each network for processing
            future_to_path = {}
            for path in df["path"]:
                # Submit network processing job with inner parallelism
                future = executor.submit(
                    process_network,
                    path,
                    index_fn,
                    max_links_in_disruption=args.max_links_ratio,
                    max_disruption_scenarios=args.max_scenarios,
                    workers=args.inner_workers,
                    show_progress=False,
                )
                future_to_path[future] = path

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                network_id = Path(path).parent.name

                try:
                    result = future.result()
                    if result:
                        output_file = path_to_save / f"{network_id}.json"
                        with open(output_file, "w") as f:
                            json.dump(result, f)

                        completed_networks += 1
                        logger.info(f"Completed {network_id} ({completed_networks}/{total_networks})")
                except Exception as e:
                    failed_networks += 1
                    logger.error(f"Error processing {network_id}: {str(e)}")

    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully processed: {completed_networks}/{total_networks} networks")
        if failed_networks > 0:
            logger.warning(f"Failed to process: {failed_networks} networks")


if __name__ == "__main__":
    main()

