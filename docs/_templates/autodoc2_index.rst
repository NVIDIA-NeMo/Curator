API Reference
=============

NeMo Curator's API reference provides comprehensive technical documentation for all modules, classes, and functions. Use these references to understand the technical foundation of NeMo Curator and integrate it with your data curation workflows.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`server;1.5em;sd-mr-1` Execution Backends
      :link: backends/backends
      :link-type: doc
      :class-card: sd-border-0

      **Ray-based execution backends**

      Adapters and executors for running pipelines at scale.
      
      :bdg-secondary:`ray-data` :bdg-secondary:`xenna`

   .. grid-item-card:: :octicon:`workflow;1.5em;sd-mr-1` Pipeline
      :link: pipeline/pipeline
      :link-type: doc
      :class-card: sd-border-0

      **Orchestrate end-to-end workflows**

      Build and run pipelines composed of processing stages.

   .. grid-item-card:: :octicon:`stack;1.5em;sd-mr-1` Processing Stages
      :link: stages/stages
      :link-type: doc
      :class-card: sd-border-0

      **Download, transform, and write data**

      Modular stages for download/extract, text models/classifiers, I/O, and utilities.
      
      :bdg-secondary:`download` :bdg-secondary:`text` :bdg-secondary:`io` :bdg-secondary:`modules`

   .. grid-item-card:: :octicon:`tasklist;1.5em;sd-mr-1` Tasks
      :link: tasks/tasks
      :link-type: doc
      :class-card: sd-border-0

      **Core data structures**

      Document batches, file groups, and related interfaces passed between stages.

   .. grid-item-card:: :octicon:`gear;1.5em;sd-mr-1` Utilities
      :link: utils/utils
      :link-type: doc
      :class-card: sd-border-0

      **Helper functions**

      File, performance, and operation utilities used across the pipeline.

.. toctree::
   :maxdepth: 1
   :caption: API Modules
   :hidden:

   backends/backends
   pipeline/pipeline
   stages/stages
   tasks/tasks
   utils/utils
