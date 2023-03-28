package it.unibo.andrp

import model.DataPoint
import config.AlgorithmConfig

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

package object dataset {
    def getDataPoints(csvDataset: DataFrame): (RDD[DataPoint], RDD[DataPoint]) = {
        val rddDataset = csvDataset.rdd.map(_fromDataFrameRowToRDD)
        splitData(rddDataset, AlgorithmConfig.TRAINING_RATIO, AlgorithmConfig.SEED)
    }

    private def _fromDataFrameRowToRDD(row: Row): DataPoint = {
        val features = row.toSeq.dropRight(1).map(_.toString.toDouble)
        val label = row.toSeq.last.toString.toDouble
        DataPoint(features, label)
    }

    private def splitData(data: RDD[DataPoint], trainingRatio: Double, seed: Long): (RDD[DataPoint], RDD[DataPoint]) = {
        val splits = data.randomSplit(Array(trainingRatio, 1.0 - trainingRatio), seed)
        (splits(0), splits(1))
    }
}
