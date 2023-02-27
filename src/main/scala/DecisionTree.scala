package it.unibo.andrp

import scala.util.Random
import breeze.linalg.DenseVector

class DecisionTree(val maxDepth: Int, val numFeatures: Int) {

  private var root: Option[Node] = None
  private var features: List[Int] = Nil
  private var taken: List[Int] = Nil

  def train(data: List[DataPoint], features: List[Int]): Unit = {
    this.features = features
    this.taken = features
    root = Some(buildTree(data, 0))
  }

  def predict(dataPoint: DataPoint): Int = {
    root match {
      case Some(node) => node.predict(dataPoint)
      case None => throw new Exception("Tree has not been trained yet")
    }
  }

  private def buildTree(data: List[DataPoint], depth: Int): Node = {
    if (depth >= maxDepth || data.isEmpty || isHomogeneous(data) || features.size == 0) {
      Leaf(getMajorityClass(data))
    } else {
      val (left, right, feature, threeshold) = splitData(data)
      if (left.isEmpty || right.isEmpty) {
        Leaf(getMajorityClass(data))
      } else {

        InternalNode(feature, threeshold, buildTree(left, depth + 1), buildTree(right, depth + 1))
      }
    }
  }

  private def selectFeature(data: List[DataPoint]): Int = {
    val max_taken_impurity_feature = taken.maxBy(informationGain(data, _))
    //val index = Random.nextInt(taken.length)
    val index_taken = features.indexOf(max_taken_impurity_feature)
    val feature = taken(index_taken)
    taken = taken.take(index_taken) ++ taken.drop(index_taken + 1)
    index_taken
  }

  private def isHomogeneous(data: List[DataPoint]): Boolean =
    data.map(_.label).distinct.length <= 1

  private def getMajorityClass(data: List[DataPoint]): Int =
    data.groupBy(_.label).mapValues(_.size).maxBy(_._2)._1

  private def splitData(data: List[DataPoint]): (List[DataPoint], List[DataPoint], Int, Double) = {

    /*val remainingFeatures = features.filterNot(_ == bestFeature)
    val featureValues = data.map(_.features(bestFeature)).distinct
    val children = featureValues.map { value =>
      val subset = data.filter(dp => dp.features(bestFeature) == value)
      val childNode = buildTree(subset, depth - 1, remainingFeatures)
      (value, childNode)
    }*/

    val index_feature = selectFeature(data)
    val threshold = findBestThreshold(data, features(index_feature), impurity)
    val (left, right) = data.partition(_.features(index_feature) <= threshold)
    (left, right, features(index_feature), threshold)
  }

  private def splitThreshold(data: List[DataPoint], feature: Int): Double = {
    val featureValues = data.map(_.features(feature))
    (featureValues.min + featureValues.max) / 2.0
  }

  def findBestThreshold(data: List[DataPoint], featureIndex: Int, impurityFunc: List[Int] => Double): Double = {
    val sortedData = data.sortBy(_.features(featureIndex))
    var bestThreshold = 0.0
    var bestGain = 0.0
    for (i <- 1 until sortedData.length) {
      if (sortedData(i - 1).features(featureIndex) != sortedData(i).features(featureIndex)) {
        val threshold = (sortedData(i - 1).features(featureIndex) + sortedData(i).features(featureIndex)) / 2.0
        val (left, right) = sortedData.splitAt(i)
        val gain = impurityFunc(left.map(_.label)) * left.length / data.length + impurityFunc(right.map(_.label)) * right.length / data.length
        if (gain > bestGain) {
          bestGain = gain
          bestThreshold = threshold
        }
      }
    }
    bestThreshold
  }

  def accuracy(dataset: List[DataPoint]): Double = {
    val predictions = dataset.map(this.predict)
    val correct = predictions.zip(dataset).count { case (pred, dp) => pred == dp.label }
    correct.toDouble / dataset.length.toDouble
  }

  sealed trait Node {
    def predict(dataPoint: DataPoint): Int


  }

  case class Leaf(label: Int) extends Node {

    def predict(dataPoint: DataPoint): Int = label
  }

  case class InternalNode(feature: Int, threshold: Double, left: Node, right: Node) extends Node {
    def predict(dataPoint: DataPoint): Int =
      if (dataPoint.features(feature) <= threshold) left.predict(dataPoint) else right.predict(dataPoint)
  }


  def printTree(node: Node, indent: String = ""): Unit = {
    node match {
      case Leaf(label) => println(s"$indent Prediction: $label")
      case InternalNode(feature, threshold, left, right) =>
        println(s"$indent Feature $feature < $threshold")
        printTree(left, indent + "    ")
        println(s"$indent Feature $feature >= $threshold")
        printTree(right, indent + "    ")
    }
  }

  def printDecisionTree(): Unit = {
    println("Decision Tree")
    println("-------------")
    println(s"Maximum depth: ${this.maxDepth}")
    println(s"Number of features: ${this.numFeatures}")
    println("")

    printTree(this.root.get)
  }

  def impurity(labels: List[Int]): Double = {
    val counts = labels.groupBy(identity).mapValues(_.size)
    val proportions = counts.values.map(_.toDouble / labels.size)
    -proportions.map(p => if (p == 0) 0 else p * math.log(p)).sum
  }

  def informationGain(data: List[DataPoint], feature: Int): Double = {
    val labels = data.map(_.label)
    val totalImpurity = impurity(labels)
    val featureValues = data.map(dp => dp.features(feature)).distinct
    val weightedImpurities = featureValues.map { value =>
      val subset = data.filter(dp => dp.features(feature) == value)
      val proportion = subset.size.toDouble / data.size
      proportion * impurity(subset.map(_.label))
    }
    totalImpurity - weightedImpurities.sum
  }
}
