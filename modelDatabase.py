import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import closing
from ultralytics import YOLO
from pathlib import Path
 
class ModelRegistry:
    def __init__(self, db_path: str = 'model_registry.db'):
        """Initialize the ModelRegistry with a database path."""
        self.db_path = db_path
        self._init_db()
 
    def _init_db(self):
        """Initialize the database table."""
        query = '''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cvat_project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
 
                trained_at DATETIME,
                storage_path TEXT NOT NULL,
                file_size_mb REAL,
                vram_gb REAL,
               
                task TEXT NOT NULL,
                labels TEXT NOT NULL,
                primary_score REAL NOT NULL,
                primary_score_type TEXT NOT NULL,
                metrics TEXT NOT NULL,
                created_at DATETIME
            )
        '''
        with closing(self._get_connection()) as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute(query)
 
    def _get_connection(self) -> sqlite3.Connection:
        """Create a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # To access columns by name
        return conn
 
    def create_model(self,
                     cvat_project_id: int,
                     name: str,
                     version: str,
                     storage_path: str,
                     task: str,
                     labels: List[str],
                     primary_score: float,
                     primary_score_type: str,
                     metrics: Dict[str, Any],
                     file_size_mb: Optional[float] = None,
                     vram_gb: Optional[float] = None,
                     trained_at: Optional[datetime] = None) -> int:
        """
        Register a new model in the database.
        Returns the ID of the newly created model.
        """
        query = '''
            INSERT INTO models (
                cvat_project_id, name, version, trained_at, storage_path,
                file_size_mb, vram_gb, task, labels, primary_score,
                primary_score_type, metrics, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
       
        metrics_json = json.dumps(metrics)
        labels_json = json.dumps(labels)
        created_at = datetime.now()
       
        data = (
            cvat_project_id, name, version, trained_at, storage_path,
            file_size_mb, vram_gb, task, labels_json, primary_score,
            primary_score_type, metrics_json, created_at
        )
       
        with closing(self._get_connection()) as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute(query, data)
                return cursor.lastrowid
 
    def get_pt_model_info(self, model):
 
        # Model hardware info
        trained_at = model.ckpt['date']
        model_path = model.ckpt_path
        model_size_mb = os.path.getsize(model_path)/1024/1024
        vram_gb = model.info(verbose=True)[1] * 2 / 1024 / 1024 /1024
       
        # Model task info
        task = model.task
        labels = model.names
        metrics = model.ckpt['train_metrics']
 
        if task == 'classify':
            accuracy_top1 = metrics['metrics/accuracy_top1']
            primary_score, score_type = accuracy_top1, 'accuracy_top1'
           
        elif task == 'detect':
           
            mAP50 = metrics['metrics/mAP50(B)']
            primary_score, score_type = mAP50, 'mAP50(Box)'
 
        elif task == 'segment':
 
            mAP50 = metrics['metrics/mAP50(M)']
            primary_score, score_type = mAP50, 'mAP50(Mask)'
 
        information = {
            'trained_at': trained_at,
            'model_path': model_path,
            'model_size_mb': model_size_mb,
            'vram_gb': vram_gb,
            'task': task,
            'labels': labels,
            'primary_score': primary_score,
            'score_type': score_type,
            'metrics': metrics
        }
        return information
   
    def register_model(self, cvat_project_id: int, name: str, version: str, model: YOLO | Path | str):
 
        if isinstance(model, (str, Path)):
            model = YOLO(model)
        
        if isinstance(model, YOLO):
            model_info = self.get_pt_model_info(model)
        else:
            raise ValueError("Model must be a YOLO object or a path to a YOLO model.")
 
        model_id = self.create_model(
            cvat_project_id=cvat_project_id,
            name=name,
            version=version,
            storage_path=model_info['model_path'],
            task=model_info['task'],
            labels=model_info['labels'],
            primary_score=model_info['primary_score'],
            primary_score_type=model_info['score_type'],
            metrics=model_info['metrics'],
            file_size_mb=model_info['model_size_mb'],
            vram_gb=model_info['vram_gb'],
            trained_at=model_info['trained_at']
        )
        return model_id
 
    def get_model(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a model by its ID."""
        query = "SELECT * FROM models WHERE id = ?"
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (model_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
 
    def list_models(self,
                    cvat_project_id: Optional[int] = None,
                    name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List models, optionally filtering by cvat_project_id or name.
        """
        query = "SELECT * FROM models WHERE 1=1"
        params = []
       
        if cvat_project_id is not None:
            query += " AND cvat_project_id = ?"
            params.append(cvat_project_id)
       
        if name is not None:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        
        # Sort by primary score (best models first)
        query += " ORDER BY primary_score DESC"
           
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_unique_projects(self) -> List[Dict[str, Any]]:
        """
        Get unique CVAT projects that have trained models.
        Returns list of dicts with cvat_project_id and count of models.
        """
        query = """
            SELECT cvat_project_id, COUNT(*) as model_count
            FROM models
            GROUP BY cvat_project_id
            ORDER BY cvat_project_id
        """
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
 
    def update_model(self, model_id: int, **kwargs) -> bool:
        """
        Update fields of a model.
        kwargs should match column names.
        Returns True if a row was updated, False otherwise.
        """
        if not kwargs:
            return False
           
        # Handle JSON serialization for special fields
        if 'metrics' in kwargs and isinstance(kwargs['metrics'], dict):
            kwargs['metrics'] = json.dumps(kwargs['metrics'])
        if 'labels' in kwargs and isinstance(kwargs['labels'], (list, dict)):
            kwargs['labels'] = json.dumps(kwargs['labels'])
           
        set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        query = f"UPDATE models SET {set_clause} WHERE id = ?"
        params = list(kwargs.values())
        params.append(model_id)
       
        with closing(self._get_connection()) as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.rowcount > 0
 
    def delete_model(self, model_id: int) -> bool:
        """Delete a model by ID."""
        query = "DELETE FROM models WHERE id = ?"
        with closing(self._get_connection()) as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute(query, (model_id,))
                return cursor.rowcount > 0
 
if __name__ == '__main__':
    registry = ModelRegistry()
 
    model_path = Path('/home/ccy/training_job/1.Train/durian-classification/detect/v11s/weights/best.pt')
    model = YOLO(model_path)
   
    id = registry.register_model(
        cvat_project_id=1,
        name='test',
        version='1.0',
        model=model
    )
    print(id)
 
    # registry.create_model(
    #     cvat_project_id=1,
    #     name='test',
    #     version='1.0',
    #     storage_path='/path/to/model',
    #     task='detection',
    #     labels= {'0': 'person', '1': 'car'},
    #     primary_score=0.95,
    #     primary_score_type='mAP50',
    #     metrics={'precision': 0.9, 'recall': 0.8},
    #     file_size_mb=100.0,
    #     vram_gb=1.5,
    #     trained_at=datetime.now()
    # )