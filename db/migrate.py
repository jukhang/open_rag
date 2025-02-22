import os

from db.base import Base, engine
from db.models import *

def create_tables():
    Base.metadata.create_all(bind=engine)


def reset_tables():
    Base.metadata.drop_all(bind=engine)
    create_tables()
