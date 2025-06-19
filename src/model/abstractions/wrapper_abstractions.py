# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Scraping Services            *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
import os
from abc import ABC, abstractmethod
import logging
from urllib.parse import urlparse
from typing import Any, Optional, List, Dict


class AbstractWrapper(ABC):
    """
    Abstract scraping target wrapper.
    """

    def __init__(self, 
                 base_url: str,
                 auth: Any | None = None, 
                 headers: dict | None = None,
                 wait_time: float = 1.5, 
                 raw_data_path: str | None = None, 
                 logger_overwrite: Any = None) -> None:
        """
        Initiation method.
        :param base_url: Base URL.
        :param auth: Authentication data.
        :param headers: Additional headers.
        :param wait_time: Waiting time in seconds between requests or download tries.
        :param raw_data_path: Folder path to backup/output raw data.
            Defaults to None in which case raw data is not saved.
        :param logger_overwrite: Logger overwrite for logging progress.
            Defaults to None in which case the progress is not logged.
        """
        self.logger = logging.Logger(f"[Wrapper<{self}>]") if logger_overwrite is None else logger_overwrite
        self.base_url = base_url
        self.auth = auth
        self.headers = headers
        self.wait = wait_time
        self.raw_data_path = raw_data_path
        os.makedirs(name=raw_data_path, exist_ok=True)
        self.last_fetched_url = None
        self.last_fetched_response = None

    @classmethod
    @abstractmethod
    def get_source_name(cls) -> str:
        """
        Returns source name.
        :return: Source name.
        """
        pass
    
    @abstractmethod
    def get_asset_types(self) -> List[str]:
        """
        Returns available asset types.
        :return: Asset types.
        """
        pass
    
    @abstractmethod
    def get_asset_identifiers(self) -> Dict[str: List[str]]:
        """
        Returns available asset identifiers.
        :return: Asset identifiers.
        """
        pass

    @abstractmethod
    def check_connection(self, **kwargs: Optional[dict]) -> bool:
        """
        Method for checking connection.
        :param kwargs: Arbitrary keyword arguments.
        :return: True if connection was established successfully else False.
        """
        pass

    def validate_url_responsibility(self, url: str) -> bool:
        """
        Method for validating the responsibility for a URL.
        :param url: Target URL.
        :return: True, if wrapper is responsible for URL else False.
        """
        return urlparse(url).netloc in self.base_url

    @abstractmethod
    def scrape_available_asset_metadata(self, asset_type: str, callback: Any = None, start_url: str | None = None, query_params: dict | None = None) -> List[dict]:
        """
        Collects available metadata entries for a target asset type.
        :param asset_type: Asset type out of supported asset types.
        :param callback: Callback to call with collected model data batches.
            Callback should take a list of entries and arbitrary keyword arguments (**kwargs) and return true, if operation was successful else False.
        :param start_url: A starting URL for cursor pagination.
        :param query_params: Query parameters to append to next URL if missing.
        :return: List of entries of given target type.
        """
        pass
    
    @abstractmethod
    def scrape_single_asset_metadata(self, asset_type: str, identifier: str, value: str) -> dict:
        """
        Abstract method for acquiring available metadata entries for a target asset type.
        :param asset_type: Asset type out of supported asset types.
        :param identifier: Identifier type, available through wrapper.
        :param value: identifier value to identify target asset.
        :return: Asset metadata.
        """
        pass

    @abstractmethod
    def safely_fetch_response(self, url: str, params: dict | None = None, current_try: int = 3, max_tries: int = 3) -> dict:
        """
        Method for fetching response.
        :param url: Target URL.
        :param params: Request query params.
            Defaults to None.
        :param current_try: Current try.
            Defaults to 3, which results in a single fetching try with max_tries at 3.
        :param max_tries: Maximum number of tries.
            Defaults to 3.
        :return: Fetched data or empty dictionary.
        """
        pass
    
    @abstractmethod
    def compute_asset_url(self, 
                             asset_type: str,
                             asset_data: dict) -> str | None:
        """
        Computes an asset URL.
        :param asset_type: Asset type.
        :param asset_data: Asset data.
        :return: Identifying URL of the asset or None in case of failure.
        """
        pass

    @abstractmethod
    def compute_update(self, 
                       asset_type: str,
                       reference_data: dict, 
                       update_data: dict) -> dict:
        """
        Computes an update for a given asset.
        :param asset_type: Asset type.
        :param reference_data: Reference data.
        :param update_data: Update entry data.
        :return: Updated data.
        """
        pass

    @abstractmethod
    def download_asset(self, asset_type: str, asset_url: str, output_path: str) -> None:
        """
        Abstract method for downloading an asset.
        :param asset_type: Asset type.
        :param asset_url: Asset URL.
        :param output_path: Output path.
        """
        pass
