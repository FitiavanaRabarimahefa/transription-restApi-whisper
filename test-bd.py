import psycopg2
from psycopg2 import OperationalError

try:
    conn = psycopg2.connect(
        dbname="bot-rag",
        user="postgres",
        password="fitiavana",
        host="localhost",
        port="5432"
    )
    print("Connexion réussie !")
except OperationalError as e:
    print(f"Erreur de connexion à la base de données : {e}")
finally:
    if 'conn' in locals() and conn:
        conn.close()
