# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import os
from pathlib import Path

import boto3


def main(args):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="eu-west-3",
    )

    output = Path(args.output)
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=args.bucket, Prefix=args.prefix)

    count = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_path = output / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {key} ({obj['Size'] / 1024:.1f} KB)")
            s3.download_file(args.bucket, key, str(local_path))
            count += 1

    print(f"\nDone: {count} files downloaded to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all files from S3 bucket")
    parser.add_argument("--bucket", type=str, default="test-engine-capture")
    parser.add_argument("--prefix", type=str, default="detections/", help="S3 key prefix filter")
    parser.add_argument("--output", type=str, default="./s3_download", help="Local output directory")
    args = parser.parse_args()
    main(args)
