val scala321 = "3.2.1"
val sparkVersion = "3.2.1"

ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := scala321

lazy val root = (project in file("."))
  .settings(
      organization := "it.unibo",
      name := "australia-next-day-rain-prediction",
      idePackagePrefix := Some("it.unibo.andrp"),
      libraryDependencies ++= Seq(
          ("org.apache.spark" %% "spark-core" % sparkVersion).cross(CrossVersion.for3Use2_13),
          ("org.apache.spark" %% "spark-sql" % sparkVersion).cross(CrossVersion.for3Use2_13),
          ("org.apache.spark" %% "spark-mllib" % sparkVersion).cross(CrossVersion.for3Use2_13),
          "com.github.nscala-time" %% "nscala-time" % "2.32.0",
          "org.rogach" %% "scallop" % "4.1.0"
      ),
      Compile / resourceDirectory := baseDirectory.value / "data",
      Compile / mainClass := Some("it.unibo.andrp.Main"),
      assembly / mainClass := Some("it.unibo.andrp.Main"),
      assembly / assemblyJarName := "andrp.jar",
      assembly / assemblyMergeStrategy := {
          case PathList("META-INF", xs@_*) => MergeStrategy.discard
          case x => MergeStrategy.first
      },
      artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) => "andrp.jar" }
  )