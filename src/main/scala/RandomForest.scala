package it.unibo.andrp

import shaded.parquet.it.unimi.dsi.fastutil.longs.LongLists.EmptyList

import scala.util.Random


case class DataPoint(features: List[Double], label: Double)

class RandomForest(numTrees: Int, maxDepth: Int) {

  private val trees: List[DecisionTreeGB] =
    List.fill(numTrees)(new DecisionTreeGB(maxDepth, featureSubsetStrategy="sqrt"))

  def train(data: List[DataPoint]): Unit = {
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



}

object RandomForest {
  def apply(numTrees: Int, maxDepth: Int): RandomForest =
    new RandomForest(numTrees, maxDepth)
}







