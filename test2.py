#!/usr/bin/env python
# coding: utf-8

# In[6]:
import time
time.sleep(100)

import sys
from random import random
from operator import add

from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType


if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """
    spark = SparkSession        .builder        .appName("PythonPi")        .getOrCreate()
    data = [('Category A', 100, "https:// This is category A"),
        ('Category B', 120, "@ ! This is category B"),
        ('Category C', 150, "This is category C !!")]
    schema = StructType([
    StructField('Category', StringType(), True),
    StructField('Count', IntegerType(), True),
    StructField('Comment', StringType(), True)])

    # Convert list to RDD
    rdd = spark.sparkContext.parallelize(data)

    # Create data frame
    rdd1 = spark.createDataFrame(rdd,schema)
    rdd1.show()
    spark.stop()

