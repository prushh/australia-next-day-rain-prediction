package it.unibo.andrp

import shaded.parquet.it.unimi.dsi.fastutil.longs.LongLists.EmptyList

import scala.util.Random


case class DataPoint(features: List[Double], label: Int)

class RandomForest(numTrees: Int, maxDepth: Int, numFeatures: Int) {

  private val trees: List[DecisionTree] =
    List.fill(numTrees)(new DecisionTree(maxDepth, numFeatures))

  def train(data: List[DataPoint]): Unit = {
    val subFeatures = List.range(0, numFeatures)
    trees.foreach(tree => {
      val subTreeFeatures = Random.shuffle(subFeatures).take(Math.sqrt(numFeatures).toInt)
      val subData = selectDataWithReplacement(data)
      tree.train(subData, subTreeFeatures)
    })
  }

  def predict(dataPoint: DataPoint): Int = {
    val predictions = trees.map(_.predict(dataPoint)
    )
    val counts = predictions.groupBy(identity).mapValues(_.size
    )
    counts.maxBy(_._2)._1
  }

  private def selectDataWithReplacement(data: List[DataPoint]): List[DataPoint] =
    List.fill(data.length)(data(Random.nextInt(data.length)))

}

object RandomForest {
  def apply(numTrees: Int, maxDepth: Int, numFeatures: Int): RandomForest =
    new RandomForest(numTrees, maxDepth, numFeatures)
}







