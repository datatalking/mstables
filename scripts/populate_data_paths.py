#!/usr/bin/env python3
"""
Populate Data Paths

Populates the financial_data_paths table with paths discovered from the test results.
This helps consolidate data sources and prevent duplicate imports.

Author: Data Consolidation System
Version: 1.0.0
"""

import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPathPopulator:
    """Populate the financial_data_paths table with discovered paths."""
    
    def __init__(self, db_path: str = 'data/mstables.sqlite'):
        self.db_path = db_path
        self.init_table()
        
        # Load network devices from config file (required - no hardcoded defaults for security)
        self.network_devices = {}
        config_path = Path('config/machine_config.json')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'network_devices' in config:
                        self.network_devices = config['network_devices']
                    else:
                        logger.warning(f"No 'network_devices' key found in config file: {config_path}")
            except Exception as e:
                logger.error(f"Could not load config file {config_path}: {e}")
                logger.info(f"Copy config/machine_config.json.template to {config_path} and configure your machines")
        else:
            logger.warning(f"Config file not found: {config_path}")
            logger.info(f"Copy config/machine_config.json.template to {config_path} and configure your machines")
        
        # Paths should be loaded from environment variables or config file
        # Load paths from .env or config file instead of hardcoding
        self.discovered_paths = []
        env_paths = os.getenv('FINANCIAL_DATA_PATHS', '')
        if env_paths:
            try:
                self.discovered_paths = json.loads(env_paths) if env_paths.startswith('[') else env_paths.split(',')
            except:
                self.discovered_paths = [p.strip() for p in env_paths.split(',') if p.strip()]
        
        # If no paths from env, load from config file
        if not self.discovered_paths:
            config_path = Path('config/data_paths.json')
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'paths' in config:
                            self.discovered_paths = config['paths']
                except Exception as e:
                    logger.warning(f"Could not load data paths config: {e}")
    
    def init_table(self):
        """Initialize the financial_data_paths table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS financial_data_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    machine_id TEXT,
                    machine_name TEXT,
                    machine_ip TEXT,
                    username TEXT,
                    path_type TEXT,
                    file_count INTEGER DEFAULT 0,
                    total_size_bytes INTEGER DEFAULT 0,
                    file_extensions TEXT,
                    last_modified TIMESTAMP,
                    import_status TEXT DEFAULT 'not_imported',
                    import_date TIMESTAMP,
                    import_errors TEXT,
                    data_source TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_path ON financial_data_paths(path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_machine_id ON financial_data_paths(machine_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_import_status ON financial_data_paths(import_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_source ON financial_data_paths(data_source)")
            
            conn.commit()
            conn.close()
            logger.info("Financial data paths table initialized")
            
        except Exception as e:
            logger.error(f"Error initializing table: {e}")
    
    def _determine_path_type(self, path: str) -> str:
        """Determine the type of data path."""
        path_lower = path.lower()
        
        if any(keyword in path_lower for keyword in ['csv', '.csv']):
            return 'csv'
        elif any(keyword in path_lower for keyword in ['excel', 'xlsx', 'xls']):
            return 'excel'
        elif any(keyword in path_lower for keyword in ['database', 'db', 'sqlite']):
            return 'database'
        elif any(keyword in path_lower for keyword in ['json', '.json']):
            return 'json'
        elif any(keyword in path_lower for keyword in ['yahoo', 'sp500']):
            return 'yahoo_finance'
        elif any(keyword in path_lower for keyword in ['morningstar']):
            return 'morningstar'
        elif any(keyword in path_lower for keyword in ['wsj', 'wall street']):
            return 'wsj'
        elif any(keyword in path_lower for keyword in ['crypto', 'bitcoin']):
            return 'crypto'
        elif any(keyword in path_lower for keyword in ['commodities', 'metals']):
            return 'commodities'
        elif any(keyword in path_lower for keyword in ['bonds', 'fixed_income']):
            return 'bonds'
        elif any(keyword in path_lower for keyword in ['forex', 'currency']):
            return 'forex'
        else:
            return 'unknown'
    
    def _determine_data_source(self, path: str) -> str:
        """Determine the data source from the path."""
        path_lower = path.lower()
        
        if 'yahoo' in path_lower or 'sp500' in path_lower:
            return 'yahoo_finance'
        elif 'morningstar' in path_lower:
            return 'morningstar'
        elif 'wsj' in path_lower:
            return 'wsj'
        elif 'alpha_vantage' in path_lower or 'alphavantage' in path_lower:
            return 'alpha_vantage'
        elif 'polygon' in path_lower:
            return 'polygon'
        elif 'tiingo' in path_lower:
            return 'tiingo'
        elif 'quandl' in path_lower:
            return 'quandl'
        elif 'cboe' in path_lower:
            return 'cboe'
        elif 'sec' in path_lower:
            return 'sec'
        elif 'data_jane' in path_lower:
            return 'data_jane'
        elif 'global_finance' in path_lower:
            return 'global_finance'
        else:
            return 'unknown'
    
    def add_path(self, path: str, machine_info: Dict, scan_files: bool = True) -> int:
        """Add a financial data path to the tracking table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if path already exists
            cursor.execute("SELECT id FROM financial_data_paths WHERE path = ?", (path,))
            existing = cursor.fetchone()
            
            if existing:
                logger.info(f"Path already exists: {path}")
                return existing[0]
            
            # Get file information if requested
            file_count = 0
            total_size = 0
            extensions = set()
            last_modified = None
            
            if scan_files and os.path.exists(path):
                if os.path.isfile(path):
                    file_count = 1
                    total_size = os.path.getsize(path)
                    extensions.add(Path(path).suffix.lower())
                    last_modified = datetime.fromtimestamp(os.path.getmtime(path))
                elif os.path.isdir(path):
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                file_count += 1
                                total_size += os.path.getsize(file_path)
                                extensions.add(Path(file).suffix.lower())
                                
                                # Get latest modification time
                                file_mtime = os.path.getmtime(file_path)
                                if not last_modified or file_mtime > last_modified.timestamp():
                                    last_modified = datetime.fromtimestamp(file_mtime)
                                    
                            except (OSError, PermissionError):
                                continue
            
            # Determine path type
            path_type = self._determine_path_type(path)
            
            # Determine data source
            data_source = self._determine_data_source(path)
            
            # Insert the path
            cursor.execute("""
                INSERT INTO financial_data_paths (
                    path, machine_id, machine_name, machine_ip, username,
                    path_type, file_count, total_size_bytes, file_extensions,
                    last_modified, data_source, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                path,
                machine_info.get('id'),
                machine_info.get('name'),
                machine_info.get('ip'),
                machine_info.get('username'),
                path_type,
                file_count,
                total_size,
                json.dumps(list(extensions)),
                last_modified,
                data_source,
                machine_info.get('notes', '')
            ))
            
            path_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Added path {path} with {file_count} files ({total_size:,} bytes)")
            return path_id
            
        except Exception as e:
            logger.error(f"Error adding path {path}: {e}")
            return None
    
    def populate_all_paths(self):
        """Populate the database with all discovered paths."""
        logger.info("Starting to populate financial data paths...")
        
        total_added = 0
        
        # Add paths for each machine
        for machine_id, machine_info in self.network_devices.items():
            logger.info(f"Processing machine: {machine_info['name']} ({machine_id})")
            
            # Add paths for each username on this machine
            for username in machine_info['usernames']:
                machine_data = {
                    'id': machine_id,
                    'name': machine_info['name'],
                    'ip': machine_info['ip'],
                    'username': username,
                    'notes': machine_info['description']
                }
                
                # Add each discovered path
                for path in self.discovered_paths:
                    # Skip gateway device for data paths
                    if machine_id == 'gateway':
                        continue
                    
                    # Add the path
                    path_id = self.add_path(path, machine_data, scan_files=False)
                    if path_id:
                        total_added += 1
        
        logger.info(f"Added {total_added} paths to the database")
        return total_added
    
    def get_summary(self) -> Dict:
        """Get a summary of the populated data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get summary statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_paths,
                    COUNT(DISTINCT machine_id) as total_machines,
                    COUNT(DISTINCT data_source) as total_sources,
                    SUM(file_count) as total_files,
                    SUM(total_size_bytes) as total_size,
                    COUNT(CASE WHEN import_status = 'not_imported' THEN 1 END) as not_imported,
                    COUNT(CASE WHEN import_status = 'imported' THEN 1 END) as imported,
                    COUNT(CASE WHEN import_status = 'failed' THEN 1 END) as failed
                FROM financial_data_paths
            """)
            
            summary = dict(zip([desc[0] for desc in cursor.description], cursor.fetchone()))
            
            # Get breakdown by data source
            cursor.execute("""
                SELECT data_source, COUNT(*) as count, SUM(file_count) as files, SUM(total_size_bytes) as size
                FROM financial_data_paths
                GROUP BY data_source
                ORDER BY count DESC
            """)
            
            sources = []
            for row in cursor.fetchall():
                sources.append({
                    'data_source': row[0],
                    'count': row[1],
                    'files': row[2],
                    'size': row[3]
                })
            
            # Get breakdown by machine
            cursor.execute("""
                SELECT machine_name, COUNT(*) as count, SUM(file_count) as files, SUM(total_size_bytes) as size
                FROM financial_data_paths
                GROUP BY machine_name
                ORDER BY count DESC
            """)
            
            machines = []
            for row in cursor.fetchall():
                machines.append({
                    'machine_name': row[0],
                    'count': row[1],
                    'files': row[2],
                    'size': row[3]
                })
            
            conn.close()
            
            return {
                'summary': summary,
                'by_source': sources,
                'by_machine': machines
            }
            
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {}
    
    def export_to_csv(self, filename: str = 'financial_data_paths.csv'):
        """Export all paths to a CSV file."""
        try:
            import pandas as pd
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM financial_data_paths", conn)
            conn.close()
            
            df.to_csv(filename, index=False)
            logger.info(f"Exported {len(df)} paths to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")

def main():
    """Main function to populate the database."""
    print("Starting Data Path Population...")
    
    # Create populator instance
    populator = DataPathPopulator()
    
    # Populate all paths
    total_added = populator.populate_all_paths()
    
    # Get and display summary
    summary = populator.get_summary()
    
    print("\n" + "="*60)
    print("DATA PATH POPULATION SUMMARY")
    print("="*60)
    
    if summary.get('summary'):
        s = summary['summary']
        print(f"Total paths: {s['total_paths']}")
        print(f"Total machines: {s['total_machines']}")
        print(f"Total data sources: {s['total_sources']}")
        print(f"Total files: {s['total_files']:,}")
        print(f"Total size: {s['total_size']:,} bytes ({s['total_size']/1024/1024:.1f} MB)")
        print(f"Not imported: {s['not_imported']}")
        print(f"Imported: {s['imported']}")
        print(f"Failed: {s['failed']}")
    
    print("\nBy Data Source:")
    print("-" * 30)
    for source in summary.get('by_source', []):
        print(f"{source['data_source']}: {source['count']} paths, {source['files']:,} files")
    
    print("\nBy Machine:")
    print("-" * 30)
    for machine in summary.get('by_machine', []):
        print(f"{machine['machine_name']}: {machine['count']} paths, {machine['files']:,} files")
    
    # Export to CSV
    populator.export_to_csv()
    
    print(f"\nAdded {total_added} paths to database")
    print("Exported to financial_data_paths.csv")

if __name__ == "__main__":
    main() 