val scala321 = "3.2.1"
val sparkVersion = "3.2.1"

ThisBuild / version := "0.2.0-SNAPSHOT"
ThisBuild / scalaVersion := scala321

lazy val root = (project in file("."))
  .settings(
      name := "australia-next-day-rain-prediction",
      idePackagePrefix := Some("it.unibo.andrp"),
      libraryDependencies ++= Seq(
          ("org.apache.spark" %% "spark-core" % sparkVersion).cross(CrossVersion.for3Use2_13),
          ("org.apache.spark" %% "spark-sql" % sparkVersion).cross(CrossVersion.for3Use2_13),
          ("org.apache.spark" %% "spark-mllib" % sparkVersion).cross(CrossVersion.for3Use2_13),
          ("com.github.nscala-time" %% "nscala-time" % "2.32.0").cross(CrossVersion.for3Use2_13),
        ("org.scalanlp" %% "breeze" % "2.1.0").cross(CrossVersion.for3Use2_13),

        // The visualization library is distributed separately as well.
        // It depends on LGPL code
        ("org.scalanlp" %% "breeze-viz" % "2.1.0").cross(CrossVersion.for3Use2_13)

      )
)