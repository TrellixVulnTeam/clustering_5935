import pika
from time import sleep


credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(
    pika.ConnectionParameters('localhost', 5672, '/', credentials))
channel = connection.channel()

channel.queue_declare(queue='logs')

try:
    with open('./data/logs.log') as fp:
        for line in fp:
            message = line
            channel.basic_publish(exchange='',
                                  routing_key='logs',
                                  body=message)

            print('[*] message envoye : ' + message)

            # Temps entre les logs
            sleep(1)

    connection.close()
except TimeoutError:
    print('Timeout exception: La connexion avec RabbitMQ a échoué. ', TimeoutError)

connection.close()
