package it.unibo.andrp
package Utils

import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


class OutputWriter(spark: SparkSession, path: String, parallelism: Int) {

  import spark.implicits._

  private val numNode = spark.sparkContext.getExecutorMemoryStatus.size
  private var df: DataFrame = Seq.empty[(String, Double, Long, Int)].toDF("Algorithm", "Accuracy", "Time", "Run")

  def addRow(algorithm: String, accuracy: Double, time: Long, run: Int): Unit = {
    val newRow = Seq((algorithm, accuracy, time, run)).toDF("Algorithm", "Accuracy", "Time", "Run")
    df = df.union(newRow)
  }

  def saveToFile(): Unit = {
    val outputPath = s"$path/run_p${parallelism}_n${numNode}"
    val dfSingleFile = df.coalesce(1)
    dfSingleFile.write.mode(SaveMode.Overwrite).option("header", true).csv(outputPath)
  }
}
