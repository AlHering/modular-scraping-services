# -*- coding: utf-8 -*-
"""
****************************************************
*                Scraping Services                 *
*           (c) 2025 Alexander Hering              *
****************************************************

based on Modular Voice Assistant service infrastructure.
https://github.com/AlHering/modular-voice-assistant
"""
from __future__ import annotations
import os
from typing import Any
from enum import Enum
import json
import uvicorn
from pydantic import BaseModel
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi import FastAPI, APIRouter
import asyncio
from queue import Empty
import traceback
from typing import List, Dict, Generator
import traceback
from datetime import datetime as dt
from uuid import UUID
from functools import wraps
import logging
from src.model.scraping_registry_server import Worker, WorkerPackage, EndOfStreamPackage
from src.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.database.controller_data_model import populate_data_infrastructure, get_default_entries
from src.configuration import config as cfg


APP = FastAPI(title=cfg.PROJECT_NAME, version=cfg.PROJECT_VERSION,
              description=cfg.PROJECT_DESCRIPTION)
INTERFACE: WorkerRegistryServer | None = None
cfg.LOGGER = logging.getLogger("uvicorn.error")
cfg.LOGGER.setLevel(logging.DEBUG)


@APP.get("/", include_in_schema=False)
async def root() -> dict:
    """
    Redirects to Swagger UI docs.
    :return: Redirect to Swagger UI docs.
    """
    return RedirectResponse(url="/docs")


def interaction_log(func: Any) -> Any | None:
    """
    Interaction logging decorator.
    :param func: Wrapped function.
    :return: Error report if operation failed, else function return.
    """
    @wraps(func)
    async def inner(*args: Any | None, **kwargs: Any | None):
        """
        Inner function wrapper.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        requested = dt.now()
        try:
            response = await func(*args, **kwargs)
        except Exception as ex:
            response = {
                "status": "error",
                "exception": str(ex),
                "trace": traceback.format_exc()
            }
        responded = dt.now()
        log_data = {
            "request": {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            },
            "response": str(response),
            "requested": requested,
            "responded": responded
        }
        args[0].database.post_object(
            object_type="log",
            **log_data
        )
        logging_message = f"Interaction with {args[0]}: {log_data}"
        logging.info(logging_message)
        return response
    return inner


class WorkerRequest(BaseModel):
    """Config payload class."""
    worker: str
    input_package: WorkerPackage
    timeout: float | None = None


class ConfigPayload(BaseModel):
    """Config payload class."""
    worker: str
    config: dict 


class BaseResponse(BaseModel):
    """Config payload class."""
    status: str
    results: List[dict] 
    metadata: dict | None = None


class Endpoints(str, Enum):
    """
    Endpoints config.
    """
    interrupt = "interrupt"
    workers_get = "/worker/get"
    worker_process = "/worker/process"
    worker_stream = "/worker/stream"
    worker_run = "/worker/run"
    worker_reset = "/worker/reset"
    worker_stop = "/worker/stop"
    configs_get = "/configs/get"
    configs_add = "/configs/add"
    configs_patch = "/configs/patch"

    def __str__(self) -> str:
        """
        Returns string representation.
        """
        return str(self.value)


class WorkerRegistryServer(object):
    """
    Worker registry.
    """

    def __init__(self, workers: List[Worker]) -> None:
        """
        Initiation method.
        :param workers: Workers.
        """
        self.workers: Dict[str, Worker] = {worker.name: worker for worker in workers}
        self.worker_uuids = {key: None for key in self.workers}
        self.working_directory = os.path.join(cfg.DATA_FOLDER, "worker_registry")
        self.database = BasicSQLAlchemyInterface(
            working_directory=self.working_directory,
            population_function=populate_data_infrastructure,
            default_entries=get_default_entries()
        )
        self.router: APIRouter | None = None

    def setup_router(self) -> APIRouter:
        """
        Sets up an API router.
        :return: API router.
        """
        self.router = APIRouter(prefix=cfg.BACKEND_ENDPOINT_PREFIX)
        self.router.add_api_route(path="/workers/get", endpoint=self.get_workers, methods=["GET"])
        self.router.add_api_route(path="/interrupt", endpoint=self.interrupt, methods=["POST"])
        self.router.add_api_route(path="/worker/run", endpoint=self.process, methods=["POST"])
        self.router.add_api_route(path="/worker/stream", endpoint=self.process_as_stream, methods=["POST"])
        self.router.add_api_route(path="/worker/reset", endpoint=self.reset_worker, methods=["POST"])
        self.router.add_api_route(path="/worker/stop", endpoint=self.stop_worker, methods=["POST"])
        self.router.add_api_route(path="/configs/get", endpoint=self.get_configs, methods=["POST"])
        self.router.add_api_route(path="/configs/add", endpoint=self.add_config, methods=["POST"])
        self.router.add_api_route(path="/configs/patch", endpoint=self.patch_config, methods=["POST"])
        return self.router
    
    """
    Worker interaction
    """
    @interaction_log
    async def interrupt(self) -> BaseResponse:
        """
        Interrupt available workers.
        """
        tasks = [asyncio.create_task(self.reset_worker(worker=worker, config_uuid=self.worker_uuids[worker])) for worker in self.worker_uuids if self.worker_uuids[worker] is not None]
        # wait for tasks to complete
        _ = await asyncio.wait(tasks)
        return BaseResponse(status="success", results=[self.worker_uuids])

    @interaction_log
    async def get_workers(self) -> BaseResponse:
        """
        Responds available workers.
        """
        return BaseResponse(status="success", results=[self.worker_uuids])

    @interaction_log
    async def setup_and_run_worker(self, worker: str, config_uuid: str | UUID) -> BaseResponse:
        """
        Sets up and runs a worker.
        :param worker: Target worker name.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        worker = self.workers[worker]
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        try:
            if config_uuid != self.worker_uuids[worker.name] or not worker.thread.is_alive():
                entry = self.database.obj_as_dict(self.database.get_objects_by_filtermasks(object_type="worker_config", filtermasks=[FilterMask([["worker_type", "==", worker.name], ["id", "==", config_uuid]])])[0])
                worker.config = entry["config"]
                if worker.thread is not None and worker.thread.is_alive():
                    worker.reset(restart_thread=True)
                else:
                    thread = worker.to_thread()
                    thread.start()
                self.worker_uuids[worker.name] = config_uuid
            while not worker.setup_flag:
               await asyncio.sleep(.5)
            return BaseResponse(status="success", results=[{"worker": worker.name, "config_uuid": config_uuid}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"worker": worker.name, "config_uuid": config_uuid}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    @interaction_log
    async def reset_worker(self, worker: str, config_uuid: str | UUID) -> BaseResponse:
        """
        Resets a worker.
        :param worker: Target worker name.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        worker = self.workers[worker]
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        try:
            entry = self.database.obj_as_dict(self.database.get_objects_by_filtermasks(object_type="worker_config", filtermasks=[FilterMask([["worker_type", "==", worker], "id", "==", config_uuid])]))
            worker.config = entry["config"]
            worker.reset(restart_thread=True)
            while not worker.setup_flag:
                await asyncio.sleep(.5)
            worker.flush_inputs()
            worker.flush_outputs()
            self.worker_uuids[worker.name] = config_uuid
            return BaseResponse(status="success", results=[{"worker": worker.name, "config_uuid": config_uuid}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"worker": worker.name, "config_uuid": config_uuid}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    @interaction_log
    async def stop_worker(self, worker: str) -> BaseResponse:
        """
        Stops a worker.
        :param worker: Target worker name.
        :return: Response.
        """
        worker = self.workers[worker]
        try:
            if worker.thread is not None and worker.thread.is_alive():
                worker.reset()
            self.worker_uuids[worker.name] = None
            return BaseResponse(status="success", results=[{"worker": worker.name}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"worker": worker.name}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    @interaction_log
    async def process(self, worker_request: WorkerRequest) -> WorkerPackage | None:
        """
        Runs a worker process.
        :param worker_request: Worker request.
        :return: Worker package response.
        """
        worker = self.workers[worker_request.worker]
        worker.input_queue.put(worker_request.input_package)
        try:
            return worker.output_queue.get(timeout=worker_request.timeout)
        except Empty:
            return None

    def stream(self, worker_request: WorkerRequest) -> Generator[bytes, None, None]:
        """
        Runs a worker process.
        :param worker_request: Worker request.
        :return: Worker package generator.
        """
        worker = self.workers[worker_request.worker]
        worker.input_queue.put(worker_request.input_package)

        finished = False
        while not finished:
            try:
                response = worker.output_queue.get(timeout=worker_request.timeout)
                if isinstance(response, EndOfStreamPackage):
                    finished = True
                yield json.dumps(response.model_dump()).encode("utf-8")
            except Empty:
                finished = True

    @interaction_log
    async def process_as_stream(self, worker_request: WorkerRequest) -> StreamingResponse:
        """
        Runs a worker process in streamed mode.
        :param worker_request: Worker request.
        :return: Worker package response.
        """
        return StreamingResponse(self.stream(worker_request=worker_request), media_type="text/plain")

    """
    Config handling
    """
    @interaction_log
    async def add_config(self, payload: ConfigPayload) -> BaseResponse:
        """
        Adds a config to the database.
        :param worker: Target worker.
        :param config: Config.
        :return: Response.
        """
        if "id" in payload.config:
            payload.config["id"] = UUID(payload.config["id"])
        result = self.database.obj_as_dict(self.database.put_object(object_type="worker_config", worker_type=payload.worker, **payload.config))
        return BaseResponse(status="success", results=[result])
    
    @interaction_log
    async def patch_config(self, payload: ConfigPayload) -> BaseResponse:
        """
        Overwrites a config in the database.
        :param worker: Target worker type.
        :param config: Config.
        :return: Response.
        """
        if "id" in payload.config:
            payload.config["id"] = UUID(payload.config["id"])
        result = self.database.obj_as_dict(self.database.patch_object(object_type="worker_config", object_id=payload.config["id"], worker_type=payload.worker, **payload.config))
        return BaseResponse(status="success", results=[result])
    
    @interaction_log
    async def get_configs(self, worker: str | None = None) -> BaseResponse:
        """
        Retrieves configs from the database.
        :param worker: Target worker type.
            Defaults to None in which case all configs are returned.
        :return: Response.
        """
        if worker is None:
            results = [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_type(object_type="worker_config")]
        else:
            results = [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_filtermasks(object_type="worker_config", filtermasks=[FilterMask([["worker_type", "==", worker]])])]
        return BaseResponse(status="success", results=results)
    
    def __del__(self) -> None:
        """
        Deconstructs instance.
        """
        asyncio.run(self.interrupt())


"""
Backend server
"""
def run() -> None:
    """
    Runs backend server.
    """
    global APP, INTERFACE
    INTERFACE = WorkerRegistryServer(workers=[])
    APP.include_router(INTERFACE.setup_router())
    uvicorn.run("src.workers.worker_registry_server:APP",
                host=cfg.BACKEND_HOST,
                port=cfg.BACKEND_PORT,
                log_level="debug")


if __name__ == "__main__":
    run()