package it.unibo.andrp
package algorithm.tree

import algorithm.tree.DecisionTreeOld
import model.DataPoint
import config.AlgorithmConfig

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import shaded.parquet.it.unimi.dsi.fastutil.longs.LongLists.EmptyList

import scala.util.Random

class RandomForest(par : Boolean = false) {

  private val trees: List[DecisionTreeOld] =
    List.fill(AlgorithmConfig.RandomForest.NUM_TREES)(new DecisionTreeOld(par = par))

  def train(data: RDD[DataPoint]): Unit = {
    trees.foreach(tree => {
      tree.train(data,None)
    })
  }

  def predict(dataPoint: DataPoint): Double = {
    val predictions = trees.map(_.predict(dataPoint)
    )
    val counts = predictions.groupBy(identity).mapValues(_.size
    )
    counts.maxBy(_._2)._1
  }

  def accuracy(dataset: List[DataPoint]): Double = {
    val predictions = dataset.map(this.predict)
    val correct = predictions.zip(dataset).count { case (pred, dp) => pred == dp.label }
    correct.toDouble / dataset.length.toDouble
  }

}







