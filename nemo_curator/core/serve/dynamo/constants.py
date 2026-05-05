# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DEFAULT_ETCD_PORT = 2379
DEFAULT_NATS_PORT = 4222

DEFAULT_DYNAMO_NAMESPACE = "curator"
DEFAULT_DYNAMO_REQUEST_PLANE = "nats"
DEFAULT_DYNAMO_EVENT_PLANE = "nats"

ETCD_ACTOR_LABEL = "Dynamo_ETCD"
NATS_ACTOR_LABEL = "Dynamo_NATS"
FRONTEND_ACTOR_LABEL = "Dynamo_Frontend"

# Layout of the infra placement group shared by etcd, NATS, and the Dynamo frontend.
INFRA_ETCD_BUNDLE = 0
INFRA_NATS_BUNDLE = 1
INFRA_FRONTEND_BUNDLE = 2
INFRA_NUM_BUNDLES = 3

NEMO_CURATOR_DYNAMO_NAMESPACE = "nemo_curator_dynamo"
"""Ray namespace used for all Dynamo-related detached actors and placement groups.

Passed as ``namespace=`` in ``ray.init()`` from ``DynamoBackend.start()`` and
``.stop()``. Pipeline executors (Xenna, Ray Data) use their own namespace, so
there is no collision. Keeping the PGs + actors in a stable namespace lets a
reconnecting driver (server.start -> pipeline.run -> server.stop) find and
reap them across ``ray.shutdown()`` / ``ray.init()`` cycles.
"""
