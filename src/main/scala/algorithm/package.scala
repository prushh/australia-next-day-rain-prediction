package it.unibo.andrp

import model.DataPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

package object algorithm {
    def getDataPoints(csvDataset: DataFrame): RDD[DataPoint] = {
        csvDataset.rdd.map(_fromDataFrameRowToRDD)
    }

    private def _fromDataFrameRowToRDD(row: Row): DataPoint = {
        val features = row.toSeq.dropRight(1).map(_.toString.toDouble)
        val label = row.toSeq.last.toString.toDouble
        DataPoint(features, label)
    }
}
