package it.unibo.andrp
package algorithm

import algorithm.Executions.Execution
import algorithm.tree.Impurities.ImpurityFunc
import algorithm.tree.SubsetStrategies.SubsetStrategy
import config.AlgorithmConfig
import model.{DataPoint, Feature}

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

package object tree {
    def train(trainData: RDD[DataPoint], subsetStrategy: SubsetStrategy = AlgorithmConfig.DecisionTree.SUBSET_STRATEGY, weights: Option[Seq[Double]] = None)(execution: Execution): DecisionTree = {
        val trainDataArray = trainData.collect()

        val size = trainDataArray.length
        val features = trainDataArray.head.features.indices

        val featureSubset = subsetStrategy match {
            case SubsetStrategies.All => SubsetStrategies.all(features)
            case SubsetStrategies.Sqrt => SubsetStrategies.sqrt(features)
            case SubsetStrategies.Log2 => SubsetStrategies.log2(features)
        }
        val trainDataFiltered = _filterFeatures(trainData, featureSubset)

        val initWeights = _initWeights(weights, size)

        val impurity: ImpurityFunc = AlgorithmConfig.DecisionTree.IMPURITY_MEASURE match {
            case Impurities.Gini => Impurities.gini
            case Impurities.Entropy => Impurities.entropy
        }

        val tree = new algorithm.tree.DecisionTree
        tree.setRoot(
            Some(_build(trainDataFiltered, initWeights, impurity, AlgorithmConfig.DecisionTree.MAX_DEPTH, execution))
        )
        tree
    }

    def trainRandomForest(trainData: RDD[DataPoint])(execution: Execution): RandomForest = {
        val randomForest = new RandomForest()

        val trees = execution match {
            case Executions.Sequential =>
                randomForest.getTrees.map(
                    _ => {
                        train(trainData, AlgorithmConfig.RandomForest.SUBSET_STRATEGY)(execution)
                    }
                )
            case Executions.Distributed =>
                randomForest.getTrees.par.map(
                    _ => {
                        val subset = trainData.sample(withReplacement = true, 1)
                        train(subset, AlgorithmConfig.RandomForest.SUBSET_STRATEGY)(execution)
                    }
                ).toArray
        }

        randomForest.setTrees(trees)
        randomForest
    }

    private def _filterFeatures(trainData: RDD[DataPoint], featureIndices: Seq[Int]): RDD[DataPoint] = {
        trainData.map {
            point =>
                point.copy(
                    features = featureIndices.map {
                        idx => point.features(idx)
                    }.toArray
                )
        }
    }

    private def _initWeights(weights: Option[Seq[Double]], length: Int): Seq[Double] = {
        weights match {
            case Some(x) => x
            case None => Array.fill(length)(1.0)
        }
    }

    private def _build(trainData: RDD[DataPoint], weights: Seq[Double], impurity: ImpurityFunc, maxDepth: Int, execution: Execution): Node = {
        val trainDataArray = trainData.collect()

        val labels = trainDataArray.map(_.label)
        if (maxDepth == 0 || labels.length < AlgorithmConfig.DecisionTree.MIN_SPLIT_SIZE) {
            Leaf(_majorityWeighted(labels, weights))
        } else {
            val (bestFeature, bestSplits, bestGain) = execution match {
                case Executions.Sequential =>
                    val numFeatures = trainDataArray.head.features.size

                    var bestFeature = -1
                    var bestGain = Double.MinValue
                    var bestSplits = Double.MinValue

                    val labels = trainDataArray.map(_.label)

                    for (feature <- 0 until numFeatures) {
                        val values = trainDataArray.map(_.features(feature))
                        val (splits, gain) = _bestWeightedSplit(values, labels, weights, impurity)

                        if (gain > bestGain) {
                            bestFeature = feature
                            bestGain = gain
                            bestSplits = splits
                        }
                    }

                    (bestFeature, bestSplits, bestGain)
                case Executions.Distributed =>
                    val features = _generate(trainDataArray)

                    val featureValues = trainData.flatMap {
                        point =>
                            features.map {
                                feature => (feature.id, point.features(feature.id))
                            }
                    }

                    val impurityReductions = featureValues.groupByKey().flatMap {
                        case (featureId, values) =>
                            values.toArray.sorted.distinct.sliding(2).map {
                                case Array(v1, v2) =>
                                    val threshold = (v1 + v2) / 2
                                    (featureId, threshold, _impurityReduction(trainDataArray, features(featureId), weights, threshold, impurity))
                                case Array(v1) =>
                                    (featureId, v1, 0.0)
                            }
                    }

                    val res = impurityReductions.collect().maxBy(_._3)
                    res
            }

            if (bestGain == 0.0) {
                Leaf(_majorityWeighted(labels, weights))
            } else {
                val sc = SparkContext.getOrCreate()
                val (leftData, leftWeights, rightData, rightWeights) = _split(trainDataArray, weights, bestFeature, bestSplits)
                val leftChild = _build(sc.parallelize(leftData), leftWeights, impurity, maxDepth - 1, execution)
                val rightChild = _build(sc.parallelize(rightData), rightWeights, impurity, maxDepth - 1, execution)
                InternalNode(bestFeature, bestSplits, leftChild, rightChild)
            }
        }
    }

    private def _split(trainData: Seq[DataPoint], weights: Seq[Double], feature: Int, split: Double): (Seq[DataPoint], Seq[Double], Seq[DataPoint], Seq[Double]) = {
        val leftData = ArrayBuffer.empty[DataPoint]
        val leftLabels = ArrayBuffer.empty[Double]
        val leftWeights = ArrayBuffer.empty[Double]
        val rightData = ArrayBuffer.empty[DataPoint]
        val rightLabels = ArrayBuffer.empty[Double]
        val rightWeights = ArrayBuffer.empty[Double]

        for (((values, label), weight) <- trainData.map(_.features).zip(trainData.map(_.label)).zip(weights)) {
            if (values(feature) < split) {
                leftData += DataPoint(values.take(feature) ++ values.drop(feature + 1), label)
                leftLabels += label
                leftWeights += weight
            } else {
                rightData += DataPoint(values.take(feature) ++ values.drop(feature + 1), label)
                rightLabels += label
                rightWeights += weight
            }
        }
        (leftData, leftWeights, rightData, rightWeights)
    }

    private def _majorityWeighted(labels: Seq[Double], weights: Seq[Double]): Double = {
        val labelWeights = labels zip weights
        val labelCounts = labelWeights.groupBy(_._1).mapValues(_.map(_._2).sum)
        val (majorityLabel, _) = labelCounts.maxBy(_._2)
        majorityLabel
    }

    private def _bestWeightedSplit(featureValues: Seq[Double], labels: Seq[Double], weights: Seq[Double], impurity: ImpurityFunc): (Double, Double) = {
        val trainData = ((featureValues zip labels) zip weights).sortBy(_._1._1)

        val totalWeight = weights.sum
        var leftWeight = 0.0
        var leftCount = 0.0
        var rightWeight = totalWeight
        var rightCount = labels.length.toDouble
        var bestSplit = Double.MinValue
        var bestGain = Double.MinValue

        for (((value, _), weight) <- trainData.dropRight(1)) {
            leftWeight += weight
            leftCount += 1
            rightWeight -= weight
            rightCount -= 1

            if (value < trainData.head._1._1 || value > trainData.last._1._1 || value == trainData.head._1._1) {
                // Skip values that are equal to the first or last value, or that are duplicates of the first value
                // (because they would result in an empty left or right subset)
            } else {
                val impurityData = impurity(labels, weights)
                val impurityLeft = impurity(labels.take(leftCount.toInt), weights.take(leftCount.toInt))
                val impurityRight = impurity(labels.drop(leftCount.toInt), weights.drop(leftCount.toInt))

                val gain = impurityData - (leftWeight / totalWeight) * impurityLeft - (rightWeight / totalWeight) * impurityRight
                if (gain > bestGain) {
                    bestSplit = value
                    bestGain = gain
                }
            }
        }
        (bestSplit, bestGain)
    }

    private def _generate(trainData: Seq[DataPoint]): Seq[Feature] = {
        val numFeatures = trainData.head.features.length

        val featureValues = (0 until numFeatures).map { idx =>
            Feature(idx, trainData.map(_.features(idx)).distinct.sorted)
        }

        featureValues
    }

    private def _impurityReduction(trainData: Seq[DataPoint], feature: Feature, weights: Seq[Double], value: Double, impurity: ImpurityFunc): Double = {
        val dataWithWeights = trainData zip weights
        val (left, right) = dataWithWeights.partition { dp =>
            dp._1.features(feature.id) <= value
        }

        val leftWeight = left.map(_._2).sum
        val rightWeight = right.map(_._2).sum

        val impurityData = impurity(trainData.map(_.label), weights)
        val impurityLeft = impurity(left.map(_._1.label), left.map(_._2))
        val impurityRight = impurity(right.map(_._1.label), right.map(_._2))

        val impurityReduction = impurityData - (leftWeight / weights.sum) * impurityLeft - (rightWeight / weights.sum) * impurityRight
        impurityReduction
    }

    def accuracy(model: DecisionTree)(testData: Seq[DataPoint]): Double = {
        val predictions = testData.map { point =>
            _predict(model.getRoot, point)
        }

        _correct(predictions, testData)
    }

    def accuracy(model: RandomForest)(testData: Seq[DataPoint]): Double = {
        val predictions = testData.map {
            point => _predictRandomForest(model.getTrees, point)
        }

        _correct(predictions, testData)
    }

    private def _correct(preds: Seq[Double], testData: Seq[DataPoint]): Double = {
        val correct = (preds zip testData).count {
            case (pred, dp) =>
                pred == dp.label
        }
        correct.toDouble / testData.length.toDouble
    }

    private def _predict(root: Option[Node], testPoint: DataPoint): Double = {
        root match {
            case Some(node) => node.predict(testPoint)
            case None => throw new Exception("Tree has not been trained yet")
        }
    }

    private def _predictRandomForest(trees: Array[DecisionTree], testPoint: DataPoint): Double = {
        val predictions = trees.map {
            tree =>
                _predict(tree.getRoot, testPoint)
        }

        val counts = predictions.groupBy(identity).mapValues(_.length)
        counts.maxBy(_._2)._1
    }
}
