val scala321 = "3.2.1"
val sparkVersion = "3.3.1"

ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := scala321

lazy val root = (project in file("."))
  .settings(
      name := "australia-next-day-rain-prediction",
      idePackagePrefix := Some("it.unibo.andrp"),
      libraryDependencies ++= Seq(
          ("org.apache.spark" %% "spark-core" % sparkVersion).cross(CrossVersion.for3Use2_13),
          ("org.apache.spark" %% "spark-sql" % sparkVersion).cross(CrossVersion.for3Use2_13)
      )
  )