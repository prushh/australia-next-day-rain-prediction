package it.unibo.andrp

import scala.collection.mutable.ArrayBuffer

class DecisionTreeGB(val maxDepth: Int, val numFeatures: Int) {

  private var root: Option[Node] = None
  private var features: List[Int] = Nil
  private var taken: List[Int] = Nil

  def train(data: List[DataPoint], weights: Option[Seq[Double]], maxDepth: Int = 3, minSplitSize: Int = 5): Unit = {
    this.features = features
    this.taken = features
    val w = weights match {
      case Some(x) => x
      case None => List.fill(data.size)(1.0)
    }
    root = Some(buildDecisionTree(data, w, impurity, 3, 10))
  }

  def predict(dataPoint: DataPoint): Int = {
    root match {
      case Some(node) => node.predict(dataPoint)
      case None => throw new Exception("Tree has not been trained yet")
    }
  }


  def buildDecisionTree(data: Seq[DataPoint],

                        weights: Seq[Double],
                        impurityFunc: (Seq[Int], Seq[Double]) => Double,
                        maxDepth: Int,
                        minSplitSize: Int
                       ): Node = {
    val labels = data.map(_.label.toInt)
    if (maxDepth == 0 || labels.size < minSplitSize) {
      Leaf(getMajorityWeighted(labels, weights))
    } else {
      val (bestFeature, bestGain, bestSplits) = findBestFeature(data, weights, impurityFunc)
      if (bestGain == 0.0) {
        Leaf(getMajorityWeighted(labels, weights))
      } else {
        val (leftData, leftWeights, rightData, rightWeights) = splitData(data, weights, bestFeature, bestSplits)
        val leftChild = buildDecisionTree(leftData, leftWeights, impurityFunc, maxDepth - 1, minSplitSize)
        val rightChild = buildDecisionTree(rightData, rightWeights, impurityFunc, maxDepth - 1, minSplitSize)
        InternalNode(bestFeature, bestSplits, leftChild, rightChild)
      }
    }
  }


  private def splitData(
                         data: Seq[DataPoint],
                         weights: Seq[Double],
                         feature: Int,
                         split: Double
                       ): (Seq[DataPoint], Seq[Double], Seq[DataPoint], Seq[Double]) = {
    val leftData = ArrayBuffer.empty[DataPoint]
    val leftLabels = ArrayBuffer.empty[Double]
    val leftWeights = ArrayBuffer.empty[Double]
    val rightData = ArrayBuffer.empty[DataPoint]
    val rightLabels = ArrayBuffer.empty[Double]
    val rightWeights = ArrayBuffer.empty[Double]

    for (((values, label), weight) <- data.map(_.features).zip(data.map(_.label)).zip(weights)) {
      if (values(feature) < split) {
        leftData += DataPoint(values, label)
        leftLabels += label
        leftWeights += weight
      } else {
        rightData += DataPoint(values, label)
        rightLabels += label
        rightWeights += weight
      }
    }

    (leftData.toList, leftWeights.toList, rightData.toList, rightWeights.toList)
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
    val (bestFeature, bestGain, bestSplits) = findBestFeature(data, Seq(1.0,1.2), weightedImpurity)
    /*val index_feature = selectFeature(data)
    val threshold = findBestThreshold(data, features(index_feature), impurity)*/
    val (left, right) = data.partition(_.features(bestFeature) <= bestSplits)
    (left, right, bestFeature, bestSplits)
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

  private def splitThreshold(data: List[DataPoint], feature: Int): Double = {
    val featureValues = data.map(_.features(feature))
    (featureValues.min + featureValues.max) / 2.0
  }

  def findBestFeature(

                       data: Seq[DataPoint],
                       weights: Seq[Double],
                       impurityFunc: (Seq[Int], Seq[Double]) => Double
                     ): (Int, Double, Double) = {

    val numFeatures = data.head.features.size
    var bestFeature = -1
    var bestGain = -1000.0
    var bestSplits = -1000.0
    val labels = data.map(_.label)
    for (feature <- 0 until numFeatures) {
      val values = data.map(_.features(feature))

      val (gain, splits)  = findBestWeightedSplit(values, labels, weights, impurityFunc)

      if (gain > bestGain) {
        bestFeature = feature
        bestGain = gain
        bestSplits = splits
      }
    }

    (bestFeature, bestGain, bestSplits)
  }


  def findBestWeightedSplit(
                             featureValues: Seq[Double],
                             labels: Seq[Int],
                             weights: Seq[Double],
                             impurityFunc: (Seq[Int], Seq[Double]) => Double): (Double, Double) = {

    val data = featureValues.zip(labels).zip(weights).sortBy(_._1._1)
    val totalWeight = weights.sum
    var leftWeight = 0.0
    var leftCount = 0.0
    var rightWeight = totalWeight
    var rightCount = labels.length.toDouble
    var bestSplit=Double.MinValue
    var bestGain = Double.MinValue

    for (((value, label), weight) <- data.dropRight(1)) {
      leftWeight += weight
      leftCount += 1
      rightWeight -= weight
      rightCount -= 1
      if (value < data.head._1._1 || value > data.last._1._1 || value == data.head._1._1) {
        // Skip values that are equal to the first or last value, or that are duplicates of the first value
        // (because they would result in an empty left or right subset)
      } else {
        val gain = impurityFunc(labels, weights) -
          (leftWeight / totalWeight) * impurityFunc(labels.take(leftCount.toInt), weights.take(leftCount.toInt)) -
          (rightWeight / totalWeight) * impurityFunc(labels.drop(leftCount.toInt), weights.drop(leftCount.toInt))
        if (gain > bestGain) {
          bestSplit = value
          bestGain = gain
        }
      }
    }
    (bestSplit,bestGain)
  }


  def findBestWeightedSplit2(
                              featureValues: Seq[Double],
                              labels: Seq[Int],
                              weights: Seq[Double],
                              impurityFunc: (Seq[Int], Seq[Double]) => Double): (Double, Double) =  {

    // Ordina i valori della feature in ordine crescente
    val sortedValues = featureValues.zip(labels).zip(weights).sortBy(_._1._1)

    // Calcola i pesi cumulativi per le due classi
    val totalWeight = weights.sum
    var class0Weight = 0.0
    var class1Weight = weights.sum
    val cumulativeWeights = sortedValues.map { case ((value, label), weight) =>
      if (label == 0.0) {
        class0Weight += weight
      } else {
        class1Weight -= weight
      }
      (value, class0Weight, class1Weight)
    }

    // Trova il threshold ottimale utilizzando la ricerca binaria
    var bestGain = Double.NegativeInfinity
    var bestSplits = Seq.empty[(Double, Double, Double)]
    for (i <- 0 until cumulativeWeights.length - 1) {
      val (value1, class0Weight1, class1Weight1) = cumulativeWeights(i)
      val (value2, class0Weight2, class1Weight2) = cumulativeWeights(i + 1)
      val threshold = (value1 + value2) / 2.0
      val gain = impurityFunc(Seq(class0Weight1, class1Weight1), Seq(class0Weight2, class1Weight2))
      if (gain > bestGain) {
        bestGain = gain
        bestSplits = threshold
      }
    }

    (bestGain, bestSplits)
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

  def impurity(labels: Seq[Int],w:Seq[Double]): Double = {
    val counts = labels.groupBy(identity).mapValues(_.size)
    val proportions = counts.values.map(_.toDouble / labels.size)
    -proportions.map(p => if (p == 0) 0 else p * math.log(p)).sum
  }

  def informationGain(data: List[DataPoint], feature: Int): Double = {
    val labels = data.map(_.label)
    val totalImpurity = impurity(labels,Seq(1.0))
    val featureValues = data.map(dp => dp.features(feature)).distinct
    val weightedImpurities = featureValues.map { value =>
      val subset = data.filter(dp => dp.features(feature) == value)
      val proportion = subset.size.toDouble / data.size
      proportion * impurity(subset.map(_.label),Seq(1.0))
    }
    totalImpurity - weightedImpurities.sum
  }

  //gini impurity
  def weightedImpurity(labels: Seq[Int], weights: Seq[Double]): Double = {
    val weightedCounts = labels.zip(weights).groupBy(_._1).mapValues(_.map(_._2).sum)
    val totalWeight = weights.sum
    weightedCounts.values.map { count =>
      val weight = count / totalWeight
      weight * (1.0 - weight)
    }.sum
  }

  def getMajorityWeighted(labels: Seq[Int], weights: Seq[Double]): Int = {
    val labelWeights = labels.zip(weights)
    val labelCounts = labelWeights.groupBy(_._1).mapValues(_.map(_._2).sum)
    val (majorityLabel, _) = labelCounts.maxBy(_._2)
    majorityLabel
  }


}
