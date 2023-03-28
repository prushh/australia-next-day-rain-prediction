package it.unibo.andrp
package util

import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


class OutputWriter(sparkSession: SparkSession, path: String, label: String) {

    import sparkSession.implicits._

    private var df: DataFrame = Seq.empty[(String, Double, Long, Int)].toDF("Algorithm", "Accuracy", "Time", "Run")

    def addRow(algorithm: String, accuracy: Double, time: Long, run: Int): Unit = {
        val newRow = Seq((algorithm, accuracy, time, run)).toDF("Algorithm", "Accuracy", "Time", "Run")
        df = df.union(newRow)
    }

    def saveToFile(): Unit = {
        val outputPath = s"$path/run_p${label}"
        val dfSingleFile = df.coalesce(1)
        dfSingleFile.write.mode(SaveMode.Overwrite).option("header", value = true).csv(outputPath)
    }
}
