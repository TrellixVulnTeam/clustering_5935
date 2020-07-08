import pika
from clustering import IncrementalDBSCAN

dbscan = IncrementalDBSCAN()
batch = 0


# la fonction de callback
def callback(ch, method, properties, body):
    print("[x] recu %r" % body)
    features = body.decode()
    send_to_incremental_dbscan(features)
    global batch
    batch += 1


# fonction d'aiguillage des messages
def send_to_incremental_dbscan(message):
    dbscan.set_data(message)
    if batch == 20:
        print("On a 50 elements pour classifier")
        dbscan.batch_dbscan()
    if batch > 20:
        dbscan.incremental_dbscan_()


# credentiels pour la connection avec RabbitMQ
credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', 5672, '/', credentials))
channel = connection.channel()

# choix de la queue
channel.queue_declare(queue='logs')

# parametrage d'echange des messages
channel.basic_consume(queue='logs',
                      auto_ack=True,
                      on_message_callback=callback)

print(' [*] Reception des logs, CTRL+C pour quitter')
channel.start_consuming()
