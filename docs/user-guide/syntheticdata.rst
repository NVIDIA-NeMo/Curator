
.. _data-curator-syntheticdata:

======================================
Synthetic Data Generation
======================================
--------------------------------------
Background
--------------------------------------
Synthetic data generation has become increasing useful in large language model training.
It is used in pretraining, fine-tuning, and evaluation.
Synthetically generated data can be useful for adapting an LLM to low resource languages/domains, or performing knowledge distillation from other models among other purposes.
There are a variety of ways to construct synthetic data generation pipelines, with numerous LLM and classical filters.

NeMo Curator has a simple, easy-to-use set of tools that allow you to use prebuilt synthetic generation pipelines or build your own.
Any model inference service that uses the OpenAI API is compatible with the synthetic data generation module, allowing you to generate your data from any model.
Furthermore, NeMo Curator also can interface with `NeMo's Export and Deploy <https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#use-nemo-export-and-deploy-module-apis-to-run-inference>`_
module which allows you to host your own model for LLM inference.
NeMo Curator has prebuilt synthetic data generation pipelines for supervised fine-tuning (SFT) and preference data that were used to generate data for the training of `Nemotron-4 340B <https://research.nvidia.com/publication/2024-06_nemotron-4-340b>`_.
And, you can easily interweave filtering and deduplication steps in your synthetic data pipeline with the other modules in NeMo Curator.

--------------------------------------
Connecting to an LLM Service
--------------------------------------
NeMo Curator supports connecting to `OpenAI API <https://github.com/openai/openai-python?tab=readme-ov-file#openai-python-api-library>`_ compatible services and `NeMo Deploy <https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#use-nemo-export-and-deploy-module-apis-to-run-inference>`_ services.
Despite its name, the OpenAI API is used for querying models across different platforms beyond OpenAI's own models.
Here is how we can connect to `build.nvidia.com <https://build.nvidia.com/explore/discover>`_ to query Gemma 2 9b-it using NeMo Curator and the OpenAI API.

.. code-block:: python

    from openai import OpenAI
    from nemo_curator import OpenAIClient

    openai_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="<insert NVIDIA API key>",
    )
    client = OpenAIClient(openai_client)
    responses = client.query_model(
        model="mistralai/mixtral-8x7b-instruct-v0.1",
        messages=[
            {
                "role": "user",
                "content": "Write a limerick about the wonders of GPU computing.",
            }
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    print(responses[0])
    # Output:
    # A GPU with numbers in flight, Brings joy to programmers late at night.
    # With parallel delight, Solving problems, so bright,
    # In the realm of computing, it's quite a sight!

As you can see, ``OpenAIClient.query_model`` has a nearly identical function signature and behavior to the OpenAI chat completion API.
``client.query_model`` returns a list of respones, and the list is only greater than length 1 if ``n > 1`` is specified in the arugments.

The OpenAI API is great for accessing models that are hosted externally through a simple API.
However, these services are often rate limited, and if you are generating lots of synthetic data you may run into these limits.
An alternative to accessing externally hosted models is to deploy an LLM inference service yourself.
If you want to self-host models, we recommend using `NeMo's Export and Deploy <https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#use-nemo-export-and-deploy-module-apis-to-run-inference>`_ module to ensure that you get the best performance.

Assuming you deploy a model named "mistralai/mixtral-8x7b-instruct-v0.1" on your local machine following `this NeMo Deploy guide <https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#deploy-a-llm-model-to-tensorrt-llm>`_,
you can run the same query using the following code.

.. code-block:: python

    from nemo.deploy.nlp import NemoQueryLLM
    from nemo_curator import NemoDeployClient
    from nemo_curator.synthetic import Mixtral8x7BFormatter

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    nemo_client = NemoQueryLLM(url="localhost:8000", model_name=model)
    client = NemoDeployClient(nemo_client)
    repsonses = client.query_model(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Write a limerick about the wonders of GPU computing.",
            }
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        conversation_formatter=Mixtral8x7BFormatter(),
    )
    print(repsonses[0])
    # Output:
    # A GPU with numbers in flight, Brings joy to programmers late at night.
    # With parallel delight, Solving problems, so bright,
    # In the realm of computing, it's quite a sight!

Let's focus on the main differences here.

* ``nemo_client = NemoQueryLLM(url="localhost:8000", model_name=model)``. This initialization requires you to specify the model name. NemoQueryLLM is primarily built for querying a single LLM, but NeMo Curator allows you to change the model you are querying on your local server for each request.

* ``conversation_formatter=Mixtral8x7BFormatter()``. LLMs take a tokenized string of text as input, not a list of conversation turns. Therefore, during the alignment process each LLM uses a conversation format to turn the conversation into a single string. For Mixtral-8x7B-Instruct-v0.1, the format looks like this:

  .. code-block::

    <s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]

  Services that use the OpenAI API perform this formatting on the backend. In contrast, since NeMo Deploy allows you to run any model you want, you need to specify what conversation format you should use on when making the request.
  NeMo Curator provides prebuilt conversation formatters for Mixtral-8x7B-Instruct-v0.1 and Nemotron-4 340B named ``Mixtral8x7BFormatter`` and ``NemotronFormatter`` respectively.

.. note::
    OpenAI API backends likely format the conversation for you automatically. Depending on your synthetic data generation process, this may lead to incorrect results. Please refer to your service's documentation to see what kind of prompt formatting they follow.

############################
Querying a Reward Model
############################
Reward models can be used to score conversations between a user and assistant.
Instead of responding to a user prompt with text follow up as an assistant, a reward model will return a mapping of category to score.
These scores can then be used to filter the dataset to be higher quality.
Here is how we can query the Nemotron-4 340b reward model in NeMo Curator:

.. code-block:: python

    from openai import OpenAI
    from nemo_curator import OpenAIClient

    openai_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="<insert NVIDIA API key>",
    )
    client = OpenAIClient(openai_client)

    model = "nvidia/nemotron-4-340b-reward"

    messages = [
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {
            "role": "assistant",
            "content": "Ah, Paris, the City of Light! There are so many amazing things to see and do in this beautiful city ...",
        },
    ]

    rewards = client.query_reward_model(messages=messages, model=model)
    print(rewards)
    # {
    # "helpfulness": 1.6171875
    # "correctness": 1.6484375
    # "coherence": 3.3125
    # "complexity": 0.546875
    # "verbosity": 0.515625
    # }

For more details on the reward categories, please see the `Nemotron-4 340B Technical Report <https://arxiv.org/abs/2406.11704v1>`_.

--------------------------------------
Nemotron-4 340B Pipeline
--------------------------------------
Nemotron-4 340B is an LLM released by NVIDIA that synthetically generated 98% of the data used for its supervised fine-tuning and preference fine-tuning.
NeMo Curator contains prebuilt functions that allow you to follow the same process using the same prompt templates, and you can customize the pipelines to fit your usecase.

############################
Synthetic Prompt Generation
############################
Prompt generation is the process of synthetically generating the first line of a dialogue between a user and assistant.
This is also called "openline" generation.
Nemotron-4 340B used four different pipelines based on the generation of the `UltraChat dataset <https://arxiv.org/abs/2305.14233>`_ for generating open Q&A, writing, closed Q&A, and math & coding prompts.
NeMo Curator encapsulates all the synthetic data generation methods for Nemotron-4 340B in ``nemo_curator.synthetic.NemotronGenerator``.

We'll dive into all the methods it provides in the following sections, but here is a small example that establishes a pattern you will see with all of the functions.

.. code-block:: python

    from openai import OpenAI
    from nemo_curator import OpenAIClient
    from nemo_curator.synthetic import NemotronGenerator

    openai_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="<insert NVIDIA API key>"
    )
    client = OpenAIClient(openai_client)
    generator = NemotronGenerator(client)

    n_macro_topics = 20
    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    model_kwargs = {
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
    }

    responses = generator.generate_macro_topics(
        n_macro_topics=n_macro_topics, model=model, model_kwargs=model_kwargs
    )

    print(responses[0])
    # Output:
    # 1. Climate Change and Sustainable Living
    # 2. Space Exploration and the Universe
    # ...

This example should seem very similar to the ``OpenAIClient.query_model``.
We specify the model we are using just like before, along with additional keyword arguments to control the model's generation.
``generator.generate_macro_topics`` queries the LLM and asks it to generate a list of topics about the world.
There is an additional ``prompt_template`` parameter that is defaulted to the one used in Nemotron-4 340B, but it can be changed if desired.
``responses`` will be a list of responses. There will be only one response unless ``n > 1`` is specified in ``model_kwargs``.

The output of the above snippet will be a string response that contains a list of topics.
Many LLM responses in the Nemotron pipeline will contain a list.
Therefore, ``NemotronGenerator`` provides a helper function that will attempt to convert an LLM response into a Python list of strings

.. code-block:: python

    responses = generator.generate_macro_topics(
        n_macro_topics=n_macro_topics, model=model, model_kwargs=model_kwargs
    )

    topic_list = generator.convert_response_to_yaml_list(
        responses[0], model=model, model_kwargs=model_kwargs
    )
    print(topic_list[0])
    # Output:
    # Climate Change and Sustainable Living

This helper function prompts an LLM to convert the previous response into a yaml format, then attempts to parse the yaml format.
If the parsing fails, it will throw a ``YamlConversionError``.
``topic_list`` is not guaranteed to have a length of 20.
In our end to end pipelines that you will see later, NeMo Curator will raise a ``YamlConversionError`` if there is a mismatch between desired length of list and the received length of list, but this function does not check for it.

With these examples out of the way, let's look at exactly how to replicate the Nemotron-4 340B synthetic data generation pipeline in NeMo Curator.
For a more in-depth explanation of each of the steps, please refer to the `Nemotron-4 340B Technical Report <https://arxiv.org/abs/2406.11704v1>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open Q&A Prompt Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open Q&A prompt generation follows these steps:

#. Generate a list of macro topics about the world

#. Generate a list of subtopics related to each macro topic

#. Create a list of questions relating to the previously generated topics

   #. Additional topics can also be manually specified

#. Revise the questions to be more detailed

Using NeMo Curator, each step can be performed as follows:

.. code-block:: python

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    macro_topic_responses = generator.generate_macro_topics(
        n_macro_topics=20, model=model
    )
    macro_topics_list = ... # Parse responses manually or with convert_response_to_yaml_list

    subtopic_responses = generator.generate_subtopics(
        macro_topic=macro_topics_list[0], n_subtopics=5, model=model
    )
    subtopic_list = ... # Parse responses manually or with convert_response_to_yaml_list

    topics = macro_topics_list + subtopic_list

    question_responses = generator.generate_open_qa_from_topic(
        topic=topics[0], n_openlines=10, model=model
    )
    questions = ... # Parse responses manually or with convert_response_to_yaml_list

    revised_questions_responses = generator.revise_open_qa(
        openline=questions[0], n_revisions=5, model=model
    )
    revised_questions = ... # Parse responses manually or with convert_response_to_yaml_list

An end-to-end pipeline that composes all of these steps can be run with the ``NemotronGenerator.run_open_qa_pipeline``

.. code-block:: python

    open_qa_questions = generator.run_open_qa_pipeline(
        n_macro_topics=20,
        n_subtopics=5,
        n_openlines=10,
        n_revisions=5,
        model=model,
        ignore_conversion_failure=True,
    )

    print(open_qa_questions[0])
    # Output:
    # What are some effective sources of renewable energy?

This function runs all the previous steps together.
In order to do so, it tries to automatically convert the LLM responses to Python lists using ``convert_response_to_yaml_list``.
``ignore_conversion_failure=True`` will cause responses that cannot be automatically converted to be discarded instead of raising an error.
However, an error will still be thrown if the first step of the pipeline cannot be parsed successfully.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Writing Prompt Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Writing prompt generation follows these steps:

#. Generate tasks to write an email, essay, etc. about a topic

#. Revise the tasks to be more detailed

Using NeMo Curator, each step can be performed as follows:

.. code-block:: python

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    writing_tasks_responses = generator.generate_writing_tasks(
        topic="Climate Change and Sustainable Living",
        text_material_type="Poems",
        n_openlines=5,
        model=model,
    )
    writing_tasks_list = ... # Parse responses manually or with convert_response_to_yaml_list

    revised_writing_tasks_responses = generator.revise_writing_tasks(
        openline=writing_tasks_list[0], n_revisions=5, model=model
    )
    revised_writing_tasks = ...  # Parse responses manually or with convert_response_to_yaml_list

An end-to-end pipeline that composes all of these steps can be run with the ``NemotronGenerator.run_writing_pipeline``

.. code-block:: python

    writing_tasks = generator.run_writing_pipeline(
        topics=[
            "Climate Change and Sustainable Living",
            "Space Exploration and the Universe",
            ...,
        ],
        text_material_types=["Poems", "Essays", ...],
    )

    print(writing_tasks[0])
    # Output:
    # Write a poem about the most effective sources of renewable energy.

This function runs all the previous steps together.
In order to do so, it tries to automatically convert the LLM responses to Python lists using ``convert_response_to_yaml_list``.
``ignore_conversion_failure=True`` will cause responses that cannot be automatically converted to be discarded instead of raising an error.
However, an error will still be thrown if the first step of the pipeline cannot be parsed successfully.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Closed Q&A Prompt Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Closed Q&A prompt generation is simple and has a single step:

#. Given a document, generate some questions about it

Using NeMo Curator, this can be performed as follows:

.. code-block:: python

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    closed_qa_responses = generator.generate_closed_qa_instructions(
        document="Four score and seven years ago...",
        n_openlines=5,
        model=model,
    )
    closed_qa_questions = ...  # Parse responses manually or with convert_response_to_yaml_list

An end-to-end pipeline that repeats this for many documents can be run with the ``NemotronGenerator.run_closed_qa_pipeline``

.. code-block:: python

    closed_qa_questions = generator.run_closed_qa_pipeline(
        documents=["Four score and seven years ago...", ...],
        n_openlines=5,
        model=model,
    )

    print(closed_qa_questions[0])
    # Output:
    # (0, "Which President of the United States gave this speech?")

This function runs generates ``n_openlines`` questions for each document provided.
At the end, it tries to automatically convert the LLM responses to Python lists using ``convert_response_to_yaml_list``.
``ignore_conversion_failure=True`` will cause responses that cannot be automatically converted to be discarded instead of raising an error.
Unlike other pipelines, this pipeline returns a tuple of the question along with the index of the document that the question was about.
This is so that when questions are discarded if ``ignore_conversion_failure==True`` you can still know the mapping between documents and questions.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Math & Coding Prompt Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**************
Math
**************

Math prompt generation follows these steps:

#. Generate math macro topics targeted at a specific school level

#. Generate subtopics for each macro topic

#. Generate a math problem for each topic

   #. Additional topics can also be manually specified

Using NeMo Curator, each step can be performed as follows:

.. code-block:: python

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    macro_topic_responses = generator.generate_math_macro_topics(
        n_macro_topics=20,
        school_level="university",
        model=model
    )
    macro_topics_list = ... # Parse responses manually or with convert_response_to_yaml_list

    subtopic_responses = generator.generate_math_subtopics(
        macro_topic=macro_topics_list[0],
        n_subtopics=5,
        model=model
    )
    subtopic_list = ... # Parse responses manually or with convert_response_to_yaml_list

    topics = macro_topics_list + subtopic_list

    question_responses = generator.generate_math_problem(
        topic=topics[0],
        n_openlines=10,
        model=model
    )
    questions = ...  # Parse responses manually or with convert_response_to_yaml_list

An end-to-end pipeline that composes all of these steps can be run with the ``NemotronGenerator.run_math_pipeline``

.. code-block:: python

    math_questions = generator.run_math_pipeline(
        n_macro_topics=20,
        school_level="university",
        n_subtopics=5,
        n_openlines=10,
        model=model,
    )
    print(math_questions[0])
    # Output:
    # Prove that the square root of 2 is irrational.

This function runs all the previous steps together.
In order to do so, it tries to automatically convert the LLM responses to Python lists using ``convert_response_to_yaml_list``.
``ignore_conversion_failure=True`` will cause responses that cannot be automatically converted to be discarded instead of raising an error.
However, an error will still be thrown if the first step of the pipeline cannot be parsed successfully.

**************
Coding
**************

The coding generation pipeline is similar to the math generation pipeline.
Coding, in particular Python-related, prompt generation follows these steps:

#. Generate macro topics relating to Python

#. Generate subtopics for each macro topic

#. Generate a Python coding problem for each topic

   #. Additional topics can also be manually specified

Using NeMo Curator, each step can be performed as follows:

.. code-block:: python

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    macro_topic_responses = generator.generate_python_macro_topics(
        n_macro_topics=20,
        model=model
    )
    macro_topics_list = ... # Parse responses manually or with convert_response_to_yaml_list

    subtopic_responses = generator.generate_python_subtopics(
        macro_topic=macro_topics_list[0],
        n_subtopics=5,
        model=model
    )
    subtopic_list = ... # Parse responses manually or with convert_response_to_yaml_list

    topics = macro_topics_list + subtopic_list

    question_responses = generator.generate_python_problem(
        topic=topics[0],
        n_openlines=10,
        model=model
    )
    questions = ...  # Parse responses manually or with convert_response_to_yaml_list

An end-to-end pipeline that composes all of these steps can be run with the ``NemotronGenerator.run_python_pipeline``

.. code-block:: python

    python_questions = generator.run_python_pipeline(
        n_macro_topics=20,
        n_subtopics=5,
        n_openlines=10,
        model=model,
    )
    print(python_questions[0])
    # Output:
    # Demonstrate how to write a for loop in Python.

This function runs all the previous steps together.
In order to do so, it tries to automatically convert the LLM responses to Python lists using ``convert_response_to_yaml_list``.
``ignore_conversion_failure=True`` will cause responses that cannot be automatically converted to be discarded instead of raising an error.
However, an error will still be thrown if the first step of the pipeline cannot be parsed successfully.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changing Prompt Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each one of the steps above uses a prompt template that gets populated with the number of topics/openlines along with any additional information in the steps.
A prompt template in this case is just a string with a placeholder.
For example, here is the default prompt template for ``Nemotron.generate_writing_tasks``:

.. code-block:: python

    DEFAULT_WRITING_TASK_PROMPT_TEMPLATE = 'Can you generate {n_openlines} tasks, each of which requires to create a "{text_material_type}" related to {topic}? Each task should be concise and include one or two sentences only. The tasks should be as diverse as possible. Your answer should be a list of tasks.'

A complete collection of prompt templates are provided at ``nemo_curator.synthetic.prompts``.
So long as the placeholders match the required function arguments, you can swap prompt templates around.
For example, the default prompt template for generating a Python problem from a topic is ``PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE``, but it can be changed as follows.

.. code-block:: python

    from nemo_curator.synthetic import PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    macro_topic_responses = generator.generate_python_macro_topics(
        n_macro_topics=20,
        model=model
    )
    macro_topics_list = ... # Parse responses manually or with convert_response_to_yaml_list

    subtopic_responses = generator.generate_python_subtopics(
        macro_topic=macro_topics_list[0],
        n_subtopics=5,
        model=model
    )
    subtopic_list = ... # Parse responses manually or with convert_response_to_yaml_list

    topics = macro_topics_list + subtopic_list

    question_responses = generator.generate_python_problem(
        topic=topics[0],
        n_openlines=10,
        model=model,
        prompt_template=PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE,
    )
    questions = ...  # Parse responses manually or with convert_response_to_yaml_list


You can supply your own prompt template that has additional placeholders, and NeMo Curator will properly insert values for them so long as they are specified in the ``prompt_kwargs`` of the function.
For example, you can define a prompt template that generates macro topics with exceptions.

.. code-block:: python

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    my_prompt_template = "Can you generate {n_macro_topics} comprehensive topics that encompass various aspects of our daily life, the world, and science? Your answer should be a list of topics. Make the topics as diverse as possible, but do not include anything relating to {exception}"
    macro_topic_responses = generator.generate_macro_topics(
        n_macro_topics=5,
        model=model,
        prompt_template=my_prompt_template,
        prompt_kwargs={
            "exception": "illegal activities",
        },
    )

############################
Dialogue Generation
############################
After prompts are generated with the methods above and mixed together, a dialogue can be synthesized.
In the dialogue, an LLM will play the part of both user and assistant.
``Nemotron.generate_dialogue`` is a simple method to do this.

.. code-block:: python

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    dialogue = generator.generate_dialogue(
        openline="Write a poem about the moon.",
        user_model=model,
        assistant_model=model,
        n_user_turns=3,
    )
    print(dialogue)
    # Output:
    # [{"role": "user", "content": "Write a poem about the moon."},
    # {"role": "assistant", "content": "..."},
    # ...]

``n_user_turns`` specifies that there will be 3 user turns in the dialogue, where each turn is followed by 1 assistant turn.
Therefore, the total number of turns (and the length of the returned list) will always be ``2*n_user_turns``.
Having an LLM play the role of an assistant is easy, since that is what it is designed to do.
In order to impersonate a user, the following special prompt template is used:

.. code-block:: python

    DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE = "Here is a conversation between a user and an assistant.\n<|The Start of Assistant's Conversation with User|>\n{conversation_history}\n<|The End of Assistant's Conversation with User|>\n\nGiven the conversation above, generate a followup request or question in the tone of User. Directly give me the question without extraneous words."

    conversation = [
        {"role": "user", "content": "Write a poem about the moon."},
        {"role": "assistant", "content": "..."},
        ...,
    ]
    conversation_history = ""
    for turn in conversation:
        conversation_history += f"{turn['role'].capitalize()}: {turn['content']}"

    prompt = DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE.format(
        conversation_history=conversation_history
    )


#######################################
Synthetic Two-Turn Prompt Generation
#######################################
Nemotron-4 340B uses two-turn prompts for its preference data.
In this context, a two-turn prompt is a conversation that has a user turn, assistant turn, and a final user turn.
Here is an example:

.. code-block:: python

    conversation = [
        {"role": "user", "content": "Write a poem about the moon."},
        {"role": "assistant", "content": "The moon is bright. It shines at night."},
        {"role": "user", "content": "Can you make the poem longer?"},
    ]

Two-turn prompt generation is easy in NeMo Curator with ``Nemotron.generate_two_turn_prompt``.

.. code-block:: python

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    dialogue = generator.generate_two_turn_prompt(
        openline="Write a poem about the moon.",
        user_model=model,
        assistant_model=model,
    )
    print(dialogue)
    # Output:
    # conversation = [
    #    {"role": "user", "content": "Write a poem about the moon."},
    #    {"role": "assistant", "content": "The moon is bright. It shines at night."},
    #    {"role": "user", "content": "Can you make the poem longer?"},
    #]

The user impersonation follows the same format as described in the dialogue generation section.

############################
Entity Classification
############################
In addition to generating data, it can be helpful to classify a small amount of data using an LLM.
Nemotron-4 340B uses an LLM to classify Wikipedia entities to determine if they relate to math or Python progamming.
NeMo Curator provides two simple functions for classifying math and Python entities.

.. code-block:: python

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    math_classification_responses = generator.classify_math_entity(
        entity="Set theory",
        model=model,
    )
    print(math_classification_responses[0])
    # Output:
    # Yes ...

    python_classification_responses = generator.classify_python_entity(
        entity="Recipes for blueberry pie",
        model=model,
    )
    print(python_classification_responses[0])
    # Output:
    # No ...


###################################
Asynchronous Generation
###################################
All of the code so far has been sending requests to the LLM service synchronously.
This can be very ineffecient since many requests can be sent simultaneously in most of the pipelines.
Therefore, NeMo Curator provides an asynchronous alternative using OpenAI's async API.

.. code-block:: python

    from openai import AsyncOpenAI
    from nemo_curator import AsyncOpenAIClient
    from nemo_curator.synthetic import AsyncNemotronGenerator

    openai_client = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1", api_key="<insert NVIDIA API key>"
    )
    client = AsyncOpenAIClient(openai_client)
    generator = AsyncNemotronGenerator(client, max_concurrent_requests=10)

    n_macro_topics = 20
    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    model_kwargs = {
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
    }

    responses = await generator.generate_macro_topics(
        n_macro_topics=n_macro_topics, model=model, model_kwargs=model_kwargs
    )

    print(responses[0])
    # Output:
    # 1. Climate Change and Sustainable Living
    # 2. Space Exploration and the Universe
    # ...

As you can see, the asynchronous modules have the same interface as the synchronous modules.
The only exception is that a ``max_concurrent_requests`` parameter can be supplied to the constructor of ``AsyncNemotronGenerator`` as a form of rate limiting if your service is rate limited.

-----------------------------------------------
Combining with other NeMo Curator modules
-----------------------------------------------
Synthetic data generation, unlike the rest of NeMo Curator, operates independently of Dask.
This is due to the scale differences between modules.
Synthetic data is usually generated on the order of 100,000 samples while pretraining datasets operate at the scale of 1,000,000,000+ samples.
Starting up a Dask cluster for that scale is usually not needed.
However, you may want to deduplicate or filter your responses with NeMo Curator.
For example, topics might end up getting duplicated, and sending duplicate topics as queries to an LLM wastes valuable resources.

We recommend using ``DocumentDataset.from_pandas`` and ``DocumentDataset.to_pandas`` to transition between workflows that require the other NeMo Curator modules.
For example, you could do something like this:

.. code-block:: python

    import pandas as pd
    from nemo_curator.datasets import DocumentDataset

    # Initialize client, etc.

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    macro_topic_responses = generator.generate_macro_topics(
        n_macro_topics=20, model=model
    )
    macro_topics_list = ... # Parse responses manually or with convert_response_to_yaml_list

    subtopic_responses = generator.generate_subtopics(
        macro_topic=macro_topics_list[0], n_subtopics=5, model=model
    )
    subtopic_list = ... # Parse responses manually or with convert_response_to_yaml_list

    df = pd.DataFrame({"topics": subtopic_list})
    dataset = DocumentDataset.from_pandas(df)

    # Deduplicate/filter with NeMo Curator

    filtered_topics = dataset.to_pandas()["topics"].to_list()

    # Continue with synthetic data generation pipeline