package it.unibo.andrp
package algorithm.spark

import model.DataPoint
import config.AlgorithmConfig

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD


class DecisionTree extends Evaluate {
    var model: Option[DecisionTreeModel] = None

    override def fit(trainData: RDD[DataPoint]): Unit = {
        val labeledPointData = trainData.map {
            point =>
                LabeledPoint(point.label, Vectors.dense(point.features.toArray))
        }

        def mapCategoricalFeaturesInfo: Map[Int, Int] = Map(1 -> 2)

        val decisionTree = DecisionTree.trainClassifier(
            labeledPointData,
            2,
            mapCategoricalFeaturesInfo,
            "gini",
            AlgorithmConfig.DecisionTree.MAX_DEPTH,
            32
        )

        this.model = Some(decisionTree)
    }

    override def predict(testData: RDD[DataPoint]): Seq[Double] = {
        this.model match {
            case Some(x) =>
                val toPred = testData.map {
                    point => Vectors.dense(point.features.toArray)
                }
                val predictions = x.predict(toPred).collect()
                predictions
            case None => {
                println("Error, you have to train first the model!")
                Array(0.0)
            }
        }
    }


}
