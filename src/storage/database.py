from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.models.schemas import ProcessingResult, utc_now_iso


class Base(DeclarativeBase):
    pass


class ViolationRecord(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String(512), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    violation_type = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    description = Column(Text, nullable=False)
    plate_number = Column(String(32), nullable=True)
    evidence_path = Column(String(512), nullable=True)
    metadata_json = Column(Text, nullable=True)
    processing_time_ms = Column(Float, default=0.0)


class Database:
    def __init__(self, db_path: str) -> None:
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def save_result(self, result: ProcessingResult, plate_text: str | None = None) -> list[int]:
        ids: list[int] = []
        ts = datetime.fromisoformat(result.timestamp.replace("Z", ""))
        plate = plate_text or (result.plates[0].text if result.plates else None)

        with self.SessionLocal() as session:
            if not result.violations:
                record = ViolationRecord(
                    image_path=result.image_path,
                    timestamp=ts,
                    violation_type="none",
                    confidence=0.0,
                    description="No violations detected",
                    plate_number=plate,
                    evidence_path=result.evidence_path,
                    metadata_json=json.dumps({"detections": len(result.detections)}),
                    processing_time_ms=result.processing_time_ms,
                )
                session.add(record)
                session.commit()
                ids.append(record.id)
            else:
                for v in result.violations:
                    record = ViolationRecord(
                        image_path=result.image_path,
                        timestamp=ts,
                        violation_type=v.violation_type,
                        confidence=v.confidence,
                        description=v.description,
                        plate_number=plate,
                        evidence_path=result.evidence_path,
                        metadata_json=json.dumps(v.metadata),
                        processing_time_ms=result.processing_time_ms,
                    )
                    session.add(record)
                    session.commit()
                    ids.append(record.id)
        return ids

    def search(
        self,
        violation_type: str | None = None,
        plate_number: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            q = session.query(ViolationRecord)
            if violation_type:
                q = q.filter(ViolationRecord.violation_type == violation_type)
            if plate_number:
                q = q.filter(ViolationRecord.plate_number.ilike(f"%{plate_number}%"))
            rows = q.order_by(ViolationRecord.timestamp.desc()).limit(limit).all()
            return [self._row_to_dict(r) for r in rows]

    def all_records(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            rows = session.query(ViolationRecord).order_by(ViolationRecord.timestamp.desc()).limit(limit).all()
            return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: ViolationRecord) -> dict[str, Any]:
        return {
            "id": row.id,
            "image_path": row.image_path,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "violation_type": row.violation_type,
            "confidence": row.confidence,
            "description": row.description,
            "plate_number": row.plate_number,
            "evidence_path": row.evidence_path,
            "metadata": json.loads(row.metadata_json) if row.metadata_json else {},
            "processing_time_ms": row.processing_time_ms,
        }
