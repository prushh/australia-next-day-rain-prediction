package it.unibo.andrp
package SparkAlgorithm


import it.unibo.andrp.model.DataPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD


class DecisionTreeS extends MLAIalgorithms {
  var model: Option[DecisionTreeModel] = None

  override def fit(trainData: RDD[DataPoint]): Unit = {
    val labeledPointData = trainData.map(dp => LabeledPoint(dp.label, Vectors.dense(dp.features.toArray)))

    def mapCategoricalFeaturesInfo: Map[Int, Int] = (13 to 60).map(x => (x, 2)).toMap
    // Train the Decision Tree model
    val dt = DecisionTree.trainClassifier(labeledPointData, 2, mapCategoricalFeaturesInfo, "gini", 5, 32)


    this.model = Some(dt)
  }

  override def predict(testData: RDD[DataPoint]): Seq[Double] = {
    this.model match {
      case Some(x) => {
        val toPred = testData.map(dp => Vectors.dense(dp.features.toArray))
        val predictions = x.predict(toPred).collect()
        //predictions.select(col(labelCol), col("prediction"))
        predictions
      }
      case None => {
        println("Error, you have to train first the model!")
        Array(0.0)
      }
    }


    // Make predictions on the test data
  }


}
