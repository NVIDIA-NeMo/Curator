import uuid

import ray
from loguru import logger
from pylibraft.common.handle import Handle
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft_dask.common.nccl import nccl

from .ray_comms import Comms


class RAFTActor:
    """A class representing a RAFT Ray actor.

    Attributes
    ----------
    _index : int
        The index of the actor.
    _actor_name_prefix : str
        The prefix of the actor name.
    _name : str
        The name of the actor.
    _pool_size : int
        The size of the pool.
    _is_root : bool
        Whether the actor is the root.
    cb : Comms
        The communication object.
    unique_id : int
        The unique ID of the actor.
    root_unique_id : int
        The unique ID of the root actor.
    session_id : bytes
        The session ID.
    """

    def __init__(self, index: int, pool_size: int, session_id: bytes, actor_name_prefix: str = "RAFT", **kwargs):
        """Initialize the RAFT actor.

        Parameters
        ----------
        index : int
            The index of the actor.
        pool_size : int
            The size of the pool (i.e., number of actors).
        session_id : bytes
            The session ID.
        actor_name_prefix : str, optional
            The prefix of the actor name (default is "RAFT").

        Returns
        -------
        None
        """
        super().__init__(**kwargs)
        self._index = index
        self._actor_name_prefix = actor_name_prefix
        self._name = f"{self._actor_name_prefix}Actor-{self._index}"
        self._pool_size = pool_size
        self._is_root = not index
        self.cb = Comms(verbose=True, nccl_root_location="ray-actor")
        self.cb.init()
        self.unique_id = self.cb.uniqueId
        self.root_unique_id = self.unique_id if self._is_root else None
        self.session_id = session_id

    def broadcast_root_unique_id(self) -> None:
        """Broadcast the root unique ID to all actors.

        This method should only be called by the root actor.
        """
        if self._is_root:
            actor_handles = [
                ray.get_actor(name=f"{self._actor_name_prefix}Actor-{i}", namespace=None)
                for i in range(1, self._pool_size)
            ]
            futures = [actor.set_root_unique_id.remote(self.root_unique_id) for actor in actor_handles]

            # Block until all futures complete
            ray.get(futures)
        else:
            msg = "This method should only be called by the root"
            raise RuntimeError(msg)

    def _setup_nccl(self) -> None:
        """Setup NCCL communicator."""
        self._nccl = nccl()
        self._nccl.init(self._pool_size, self.root_unique_id, self._index)

    def _setup_raft(self) -> None:
        """Setup RAFT."""
        self._raft_handle = Handle(n_streams=0)

        inject_comms_on_handle_coll_only(self._raft_handle, self._nccl, self._pool_size, self._index, verbose=True)

    def _setup_post(self) -> None:
        """Setup post-processing.

        Called after setting up NCCL and RAFT, and may be used by subclasses for
        additional setup steps.
        """

    def setup(self) -> None:
        """Setup the actor.

        This method should be called after the root unique ID has been broadcast.
        """
        if self.root_unique_id is None:
            msg = "The unique ID of root is not set. Make sure `broadcast_root_unique_id` "
            "runs on the root before calling this method."
            raise RuntimeError(msg)

        try:
            logger.debug("     Setting up NCCL...")
            self._setup_nccl()
            logger.debug("     Setting up RAFT...")
            self._setup_raft()
            self._setup_post()
            logger.debug("     Setup complete!")
        except Exception as e:
            logger.error(f"An error occurred while setting up: {e}.")
            raise

    def set_root_unique_id(self, root_unique_id: int) -> None:
        """Set the root unique ID.

        Parameters
        ----------
        root_unique_id : int
            The root unique ID.

        Returns
        -------
        None
        """
        logger.debug(f"{self._name}: set_root_unique_id")
        if self.root_unique_id is None:
            self.root_unique_id = root_unique_id


def initialize_raft_actor_pool(
    pool_size: int, actor_class: RAFTActor, actor_name_prefix: str, **kwargs
) -> list[RAFTActor]:
    """Initialize a pool of RAFT actors with the specified size and configuration.

    This function initializes a pool of RAFT actors with the specified size and
    configuration. It starts the actors, sets up their communication, and returns
    a list of the initialized actors.

    Parameters
    ----------
    pool_size: int
        The number of actors to initialize in the pool.
    actor_class: RAFTActor
        The class of the RAFT actor to initialize, generally a subclass of
        `RAFTActor`.
    actor_name_prefix: str
        The prefix to use for the names of the actors in the pool.

    Returns
    -------
    pool: list[RAFTActor]
        A list of the initialized RAFT actors in the pool.
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(include_dashboard=False)

    session_id = uuid.uuid4().bytes

    # Start Actor
    pool = [
        actor_class.options(name=f"{actor_name_prefix}Actor-{i}").remote(i, pool_size, session_id=session_id, **kwargs)
        for i in range(pool_size)
    ]
    # ray.get() blocks until this completes, required before calling `setup()`
    # on non-root nodes.
    root_actor = ray.get_actor(name=f"{actor_name_prefix}Actor-0", namespace=None)
    logger.debug(f"Root actor: {root_actor}")
    ray.get(root_actor.broadcast_root_unique_id.remote())

    # Setup Comms (NCCL/Sub-communicator)
    ray.get([pool[i].setup.remote() for i in range(pool_size)])

    return pool
