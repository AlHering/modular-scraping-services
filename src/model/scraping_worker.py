# -*- coding: utf-8 -*-
"""
****************************************************
*                Scraping Services                 *
*           (c) 2025 Alexander Hering              *
****************************************************

based on Modular Voice Assistant service infrastructure.
https://github.com/AlHering/modular-voice-assistant
"""
from uuid import UUID
from threading import Thread, Event
from typing import List
import traceback
from time import sleep
from src.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.model.wrappers.civitai_api_wrapper import CivitaiAPIWrapper


WRAPPERS = {
    "civitai.com": CivitaiAPIWrapper
}


class Worker(object):
    """
    Class, representing scraping workers.
    """
    def __init__(self, 
                 database: BasicSQLAlchemyInterface,
                 wrapper_config_uuid: str | UUID,
                 target_asset_type: str,
                 target_asset_url_field: str | None = None,
                 start_url: str = "url",
                 query_params: dict | None = None) -> None:
        """
        Initiation method:
        :param database: Database handle.
        :param wrapper_config_uuid: Wrapper config UUID.
        :param target_asset_type: Target asset type.
        :param target_asset_url_field: Target asset URL field.
        :param start_url: Start URL for scraping process.
        """
        self.database = database
        self.wrapper_config_uuid = wrapper_config_uuid
        self.target_asset_type = target_asset_type
        self.target_asset_url_field = target_asset_url_field
        self.use_primary = database.primary_keys[target_asset_type] == target_asset_url_field

        self.start_url = start_url
        self.query_params = query_params

        wrapper_config = self.database.get_object_by_id("wrapper_config", self.wrapper_config_uuid)
        self.wrapper = WRAPPERS[wrapper_config.source](**wrapper_config.config)

        self.interrupt = Event()
        self.pause = Event()
        self.was_interrupted = False

        self.worker_uuid = self.database.post_object(
            object_type="worker",
            target_asset=target_asset_type,
            target_asset_url_field=target_asset_url_field,
            start_url=start_url,
            wrapper_config_uuid=wrapper_config_uuid,
            status="initiated"
        )

    def update_worker_entry(self, **updates) -> None:
        """
        Updates worker database entry.
        :param updates: Keyword based updates.
        """
        self.database.patch_object(
            object_type="worker",
            object_id=self.worker_uuid,
            **updates
        )

    def _callback(self, entries: List[dict], **kwargs: dict) -> bool:
        """
        Scraping callback method.
        :param entries: Scraped entries.
        :param kwargs: Arbitrary keyword arguments.
        """
        # handling events
        if self.interrupt.is_set():
            self.update_worker_entry(last_url=self.wrapper.last_fetched_url, status="interrupted")
            self.was_interrupted = True
            return False
        if self.pause.is_set():
            self.update_worker_entry(last_url=self.wrapper.last_fetched_url, status="paused")
            while self.pause.is_set():
                if self.interrupt.is_set():
                    self.update_worker_entry(last_url=self.wrapper.last_fetched_url, status="interrupted")
                    self.was_interrupted = True
                    return False
                sleep(10)
            self.update_worker_entry(status="running")

        # Processing entries
        for entry in entries:
            url = self.wrapper.compute_asset_url(asset_type=self.target_asset_type, asset_data=entry)
            # Search for existing entry
            if self.use_primary:
                existing_entry = self.database.get_object_by_id(object_type=self.target_asset_type,
                                                                object_id=url)
            else:
                existing_entries = self.database.get_objects_by_filtermasks(object_type=self.target_asset_type,
                                                                            filtermasks=FilterMask([[self.target_asset_url_field, "==", ""]]))
                if existing_entries:
                    existing_entry = existing_entries[0]
                else:
                    existing_entry = None

            if existing_entry:
                entry = self.wrapper.compute_update(asset_type=self.target_asset_type,
                                                    reference_data=existing_entry,
                                                    update_data=entry)
            # Patch or post entry
            self.database.put_object(object_type=self.target_asset_type,
                                     reference_attributes=[self.database.primary_keys[self.target_asset_type]],
                                     object_attributes=entry)
        return True


    def _handle_run(self) -> None:
        """
        Handles scraping run.
        """
        try:
            if self.target_asset_type not in self.wrapper.get_asset_types():
                raise ValueError(f"{type(self.wrapper).__name__} wrapper does not support asset type '{self.target_asset_type}'")
            if self.start_url and not self.wrapper.validate_url_responsibility(self.start_url):
                raise ValueError(f"{type(self.wrapper).__name__} wrapper not responsible for URL '{self.start_url}'")
            self.wrapper.scrape_available_asset_metadata(
                asset_type=self.target_asset_type,
                callback=self._callback,
                start_url=self.start_url,
                query_params=self.query_params
            )
            if not self.was_interrupted:
                self.update_worker_entry(last_url=self.wrapper.last_fetched_url, status="finished")
        except Exception as ex:
            self.database.post_object(
                object_type="scraping_error",
                worker_uuid=self.worker_uuid,
                exception=str(ex),
                trace=traceback.format_exc()
            )
            self.update_worker_entry(last_url=self.wrapper.last_fetched_url, status="error")

    def run(self) -> None:
        """
        Main runner method.
        """
        thread = Thread(target=self._handle_run)
        thread.daemon = True
        thread.run()
        thread.join()
