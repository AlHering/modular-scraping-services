# -*- coding: utf-8 -*-
"""
****************************************************
*                Scraping Services                 *
*           (c) 2025 Alexander Hering              *
****************************************************
"""
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Engine, Column, String, JSON, Integer, DateTime, func, Boolean, ForeignKey, Text
from sqlalchemy_utils import UUIDType
from uuid import uuid4


def populate_data_infrastructure(engine: Engine, schema: str, model: dict) -> None:
    """
    Function for populating data infrastructure.
    :param engine: Database engine.
    :param schema: Schema for tables.
    :param model: Model dictionary for holding data classes.
    """
    schema = str(schema)
    if schema and not schema.endswith("."):
        schema += "."
    base = declarative_base()

    class Log(base):
        """
        Log class, representing an log entry, connected to a backend interaction.
        """
        __tablename__ = f"{schema}log"
        __table_args__ = {
            "comment": "Log table.", "extend_existing": True}

        id = Column(UUIDType(binary=False), primary_key=True, unique=True, nullable=False, default=uuid4,
                    comment="UUID of the logging entry.")
        request = Column(JSON, nullable=False,
                         comment="Request, sent to the backend.")
        response = Column(JSON, comment="Response, given by the backend.")
        requested = Column(DateTime, server_default=func.now(),
                           comment="Timestamp of request receive.")
        responded = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                           comment="Timestamp of response transmission.")
        
    class WrapperConfig(base):
        """
        Config class, representing an API wrapper config.
        """
        __tablename__ = f"{schema}wrapper_config"
        __table_args__ = {
            "comment": "API wrapper config table.", "extend_existing": True}

        uuid = Column(UUIDType(binary=False), primary_key=True, unique=True, nullable=False, default=uuid4,
                    comment="UUID of an instance.")
        
        source_name = Column(String,
                         comment="Wrapper source name.")
        config = Column(JSON,
                         comment="Wrapper config.")
        validated = Column(Boolean, default=False,
                         comment="Validation flag.")
        
        workers = relationship("Worker", backref="wrapper_config")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        
    class Worker(base):
        """
        Worker class, representing an scraping worker.
        """
        __tablename__ = f"{schema}worker"
        __table_args__ = {
            "comment": "Worker table.", "extend_existing": True}

        uuid = Column(UUIDType(binary=False), primary_key=True, unique=True, nullable=False, default=uuid4,
                    comment="UUID of an instance.")
        target_asset_type = Column(String, nullable=False,
                         comment="Target asset type.")
        target_asset_url_field  = Column(String, nullable=False, default="url",
                         comment="Target asset URL field.")
        start_url = Column(String,
                         comment="Start URL config.")
        last_url = Column(String,
                         comment="Start URL config.")
        status = Column(String,
                         comment="Status out of 'initiated', 'started', 'running', 'error', 'finished'.")
        
        wrapper_config_uuid = Column(Integer, ForeignKey(f"{schema}wrapper_config.uuid"))
        wrapper_config = relationship("WrapperConfig", back_populates="workers")
        errors = relationship("ScrapingError", backref="worker")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        
    class ScrapingError(base):
        """
        Error class, representing an scraping errors.
        """
        __tablename__ = f"{schema}scraping_error"
        __table_args__ = {
            "comment": "Scraping error table.", "extend_existing": True}

        uuid = Column(UUIDType(binary=False), primary_key=True, unique=True, nullable=False, default=uuid4,
                    comment="UUID of the entry.")
        
        worker_uuid = Column(Integer, ForeignKey(f"{schema}wrapper_config.uuid"))
        worker = relationship("WrapperConfig", back_populates="workers")

        exception = Column(String, nullable=False,
                        comment="Exception text.")
        trace = Column(Text, nullable=False,
                        comment="Trace of the error.")
        created = Column(DateTime, server_default=func.now(),
                        comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), onupdate=func.now(),
                        comment="Timestamp of last update.")

    for dataclass in [Log, WrapperConfig, Worker, ScrapingError]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)


def get_default_entries() -> dict:
    """
    Returns default entries.
    :return: Default entries.
    """
    return {}