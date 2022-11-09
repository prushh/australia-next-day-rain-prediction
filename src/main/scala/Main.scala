package it.unibo.andrp

import config.SparkProjectConfig
import utils.TimestampFormatter

import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

import scala.io.Source
import scala.util.Using

@main def run(): Unit =
    /*
     * Loading Spark and Hadoop.
     */
    val sparkSession = SparkProjectConfig.sparkSession("local[*]", 1)
    val sparkContext = sparkSession.sparkContext

    val url = ClassLoader.getSystemResource("weatherAUS.json")
    val schemaSource = Source.fromFile(url.getFile)
    val schemaFromJson = DataType.fromJson(schemaSource.getLines().mkString).asInstanceOf[StructType]

    val df = sparkSession.read
      .option("header", value = true)
      .option("timestampFormatter", TimestampFormatter.timestampPattern)
      .schema(schemaFromJson)
      .csv("data/weatherAUS.csv")

    sparkSession.stop()
