Getting Started
===============

The ``takco`` package is designed to be run in several different environments. If you
are just exploring the options, you can install the package and make use of external 
APIs. This gives you access to Wikipedia or other web table sources, external Knowledge
Bases, and entity query APIs. If you want to run a larger pipeline, you should setup
a local KB, and mirror the web table sources yourself. Finally, if you want to reproduce
the research on which ``takco`` was built, you can run it on a compute cluster.

Default Setup
~~~~~~~~~~~~~

By default, ``takco`` is configured to make use of external APIs for web table 
harvesting and entity linking. This way, you can explore its features without having
to setup a large KB yourself.

...



Large Machine Setup
~~~~~~~~~~~~~~~~~~~

For larger workloads, setup a locally hosted mirror of Wikipedia.

.. warning::

    This will require some storage space. A typical Wikipedia zim dump is about 40 GB.

1. Install the `Kiwix tools <https://github.com/kiwix/kiwix-tools>`_.
2. Download a `Wikipedia zim dump <https://dumps.wikimedia.org/other/kiwix/zim/wikipedia/>`_.
3. Host it with ``./kiwix-serve --port=8989 your_wiki_dump.zim``


On a machine with many cores, it is often useful to use the `Dask <http://dask.org>`_
execution engine, which provides a dashboard for running tasks.


Installing Trident
^^^^^^^^^^^^^^^^^^

...


Cluster Setup
~~~~~~~~~~~~~

To cluster and integrate a large corpus of web tables, it is recommended to run 
``takco`` on a large cluster of machines. For this purpose, the 
`Dask <http://dask.org>`_ execution engine has several backends.

The current version of ``takco`` will be tested on SLURM.

