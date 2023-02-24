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
      val (left, right, feature) = splitData(data)
      if (left.isEmpty || right.isEmpty) {
        Leaf(getMajorityClass(data))
      } else {

        InternalNode(feature, splitThreshold(data, feature), buildTree(left, depth + 1), buildTree(right, depth + 1))
      }
    }
  }

  private def selectFeature(data: List[DataPoint]): Int = {
    val index = taken.maxBy(informationGain(data, _))
    //val index = Random.nextInt(taken.length)
    val feature = taken(index)
    taken = taken.take(index) ++ taken.drop(index + 1)
    feature
  }

  private def isHomogeneous(data: List[DataPoint]): Boolean =
    data.map(_.label).distinct.length <= 1

  private def getMajorityClass(data: List[DataPoint]): Int =
    data.groupBy(_.label).mapValues(_.size).maxBy(_._2)._1

  private def splitData(data: List[DataPoint]): (List[DataPoint], List[DataPoint], Int) = {

    /*val remainingFeatures = features.filterNot(_ == bestFeature)
    val featureValues = data.map(_.features(bestFeature)).distinct
    val children = featureValues.map { value =>
      val subset = data.filter(dp => dp.features(bestFeature) == value)
      val childNode = buildTree(subset, depth - 1, remainingFeatures)
      (value, childNode)
    }*/

    val feature = selectFeature(data)
    val threshold = splitThreshold(data, feature)
    val (left, right) = data.partition(_.features(feature) <= threshold)
    (left, right, feature)
  }

  private def splitThreshold(data: List[DataPoint], feature: Int): Double = {
    val featureValues = data.map(_.features(feature))
    (featureValues.min + featureValues.max) / 2.0
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
