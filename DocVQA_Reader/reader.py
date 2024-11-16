import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, lit, array, struct, current_timestamp, explode, size
from pyspark.sql.types import (
    StructType, StructField, StringType, BinaryType, ArrayType, TimestampType,
    LongType, DoubleType  # Changed IntegerType to DoubleType for version
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_spark_session():
    """Create and configure SparkSession with appropriate security settings."""
    conf = SparkConf()
    conf.set("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
    conf.set("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")

    return SparkSession.builder \
        .appName("DocVQA Reader") \
        .config(conf=conf) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.allowMultipleContexts", "true") \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()


class DocVQA:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(self.__class__.__name__)

    def readDataset(self, path: str) -> DataFrame:
        if not os.path.exists(path):
            raise ValueError(f"Dataset path does not exist: {path}")

        json_files = [f for f in os.listdir(path) if f.endswith('.json')]
        if not json_files:
            raise ValueError(f"No JSON files found in {path}")

        self.logger.info(f"Found {len(json_files)} JSON files in {path}")

        try:
            # Load and process each JSON file individually
            all_records = []
            for json_file in json_files:
                full_path = os.path.join(path, json_file)
                with open(full_path, 'r') as f:
                    try:
                        data = json.load(f)
                        record = self._process_file((f"file:{full_path}", json.dumps(data)))
                        self.logger.info(f"Processed {json_file}: Found {len(record['questions'])} questions")
                        all_records.append(record)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing {json_file}: {str(e)}")
                        continue

            # Convert to DataFrame
            schema = self._get_schema()
            df = self.spark.createDataFrame(all_records, schema)

            # Add debug columns
            df = df.withColumn("question_count", size(col("questions")))

            # Log statistics for debugging
            total_questions = df.select(explode("questions")).count()
            self.logger.info(f"Total questions across all files: {total_questions}")
            self.logger.info(f"Question counts by file:")
            df.select("path", "question_count").show(truncate=False)

            return df

        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            raise

    @staticmethod
    def _process_file(file: tuple) -> Dict[str, Any]:
        """
        Process a single JSON file according to the MP-DocVQA format:
        {
            "dataset_name": "MP-DocVQA",
            "dataset_version": 1.0,
            "dataset_split": "test",
            "data": [...]
        }
        """
        file_path, file_content = file
        try:
            data = json.loads(file_content)

            # Extract questions from the 'data' array
            questions = []
            question_ids = []
            doc_ids = []
            page_ids = []

            if isinstance(data, dict) and 'data' in data:
                for item in data['data']:
                    if isinstance(item, dict):
                        question = item.get('question', '').strip()
                        if question:
                            questions.append(question)
                            question_ids.append(item.get('questionId'))
                            doc_ids.append(item.get('doc_id'))
                            page_ids.append(item.get('page_ids', []))

            record = {
                "path": file_path,
                "dataset_name": data.get('dataset_name'),
                "dataset_version": float(data.get('dataset_version', 0.0)),  # Convert to float
                "dataset_split": data.get('dataset_split'),
                "questions": questions,
                "question_ids": question_ids,
                "doc_ids": doc_ids,
                "page_ids": page_ids
            }

            return record

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON from {file_path}: {str(e)}")
            return {
                "path": file_path,
                "dataset_name": None,
                "dataset_version": 0.0,  # Default float value
                "dataset_split": None,
                "questions": [],
                "question_ids": [],
                "doc_ids": [],
                "page_ids": []
            }

    @staticmethod
    def _get_schema() -> StructType:
        """Updated schema to match MP-DocVQA format with float version"""
        return StructType([
            StructField("path", StringType(), True),
            StructField("dataset_name", StringType(), True),
            StructField("dataset_version", DoubleType(), True),  # Changed to DoubleType
            StructField("dataset_split", StringType(), True),
            StructField("questions", ArrayType(StringType()), True),
            StructField("question_ids", ArrayType(LongType()), True),
            StructField("doc_ids", ArrayType(StringType()), True),
            StructField("page_ids", ArrayType(ArrayType(StringType())), True)
        ])


def main():
    try:
        spark = create_spark_session()
        reader = DocVQA(spark)
        dataset_path = "../qas"

        # Read Dataset
        df = reader.readDataset(dataset_path)

        # Show Schema and Sample Data
        print("\nDataset Schema:")
        df.printSchema()

        questions = df.select(explode("questions"))

        print("\nQuestions Count:")
        print(questions.count())

        print("\nNumber Partition:")
        print(df.rdd.getNumPartitions())

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()


if __name__ == "__main__":
    main()
