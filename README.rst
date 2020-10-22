takco
=====

🌮 takco is a modular system for extracting knowledge from tables. For example, you can use it
to extend `Wikidata <http://wikidata.org>`_ with information from Wikipedia tables.


Installing
~~~~~~~~~~

To install the newest version, download the sources and run:

.. code-block:: shell

    pip install .


Getting Started
~~~~~~~~~~~~~~~

Using the command line interface, you can run individual modules or entire pipelines.
To view the command line help, run ``takco -h``.

Example run:

.. code-block:: shell

    takco run -C config.toml your-pipeline.toml

To start using the library, check out the `tutorials <https://takco.readthedocs.io/en/latest/tutorials/intro.html>`_.