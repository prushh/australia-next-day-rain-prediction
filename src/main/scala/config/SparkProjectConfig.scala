package it.unibo.andrp
package config

import org.apache.spark.sql.SparkSession

object SparkProjectConfig {
    private def _sparkSession(master: String): SparkSession = {
        var builder = SparkSession.builder.appName("AustraliaNextDayRainPrediction")

        if (master != "default") {
            builder = builder.master(master)
        }

        builder.getOrCreate()
    }

    def sparkSession(master: String): SparkSession = {
        val session = _sparkSession(master)
        session.sparkContext.setLogLevel("WARN")

        session
    }
}
