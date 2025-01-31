from hashlib import sha256
import hashlib
from time import time
from functools import reduce

import ecdsa
import json

VERSION = 'alpha-1.0'
TARGET = 4  # mining difficulty


class BlockChain:
    def __init__(self):
        self.chain = [BlockChain.Block(0, [], 0)]
        self.pending_transactions = []

    @staticmethod
    def create_wallet():
        private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        public_key = private_key.verifying_key.to_string()
        wallet_address = hashlib.sha256(public_key).hexdigest()
        wallet_data = {
            "private_key": private_key.to_string().hex(),
            "public_key": public_key.hex(),
            "wallet_address": wallet_address
        }
        with open(input('enter filename without extension ') + '.json', "w") as file:
            json.dump(wallet_data, file)
        print(f'wallet created: {wallet_address}')

    @staticmethod
    def load_wallet(name):
        with open(f"{name}.json", "r") as file:
            wallet_data = json.load(file)
        return wallet_data

    def save_chain(self):
         with open(f"{input('enter filename to save blockchain: ')}.json", "w") as f:
                json.dump(list(map(lambda x: {'header': x.header, 'body': x.body}, self.chain)), f, indent=4)

    def load_chain(self, filename):
        with open(filename + '.json', "r") as file:
            data = json.load(file)
        chain = []
        for b in data:
            chain.append(BlockChain.Block.create_from_json(b['header'], b['body']))
        for b in chain[1:]:
            if b.get_hash()[:TARGET] != '0' * TARGET:
                raise Exception('invalid chain')
        self.chain = chain

    def add_transaction(self, sender_credentials, recipient, amount):
        data = BlockChain.load_wallet(sender_credentials)
        sender = data['wallet_address']
        public = data['public_key']
        message_hash = hashlib.sha256(f"{sender} -> {recipient}: {amount}".encode()).digest()
        private = ecdsa.SigningKey.from_string(bytes.fromhex(data["private_key"]), curve=ecdsa.SECP256k1)
        signature = private.sign(message_hash)
        self.pending_transactions.append(
            {"sender": sender, "recipient": recipient, "amount": amount, "public": public, "sign": signature.hex()})


    def get_last_hash(self):
        return self.chain[-1].get_hash()

    def list_operations(self):
        for b in self.chain:
            print(f"mining reward (25) -> {b.header['miner']}")
            for o in b.body:
                print(BlockChain.get_trans_message(o))

    def add_block(self, wallet_addr):
        b = BlockChain.Block(self.get_last_hash(), list(filter(BlockChain.verify, self.pending_transactions)), wallet_addr)
        while b.get_hash()[:TARGET] != '0' * TARGET:
            b.inc_nonce()
        self.chain.append(b)

    @staticmethod
    def get_trans_message(transaction):
        return f"{transaction['sender']} -> {transaction['recipient']}: {transaction['amount']}"

    @staticmethod
    def verify(transaction):
        public_key_hex = transaction['public']
        sign_hex = transaction['sign']
        message = BlockChain.get_trans_message(transaction)
        message_hash = sha256(message.encode()).digest()
        public_key_bytes = bytes.fromhex(public_key_hex)
        signature_bytes = bytes.fromhex(sign_hex)
        try:
            public_key = ecdsa.VerifyingKey.from_string(public_key_bytes, curve=ecdsa.SECP256k1)
            public_key.verify(signature_bytes, message_hash)
            return True
        except ecdsa.BadSignatureError:
            return False

    @staticmethod
    def find_balances(bc, addr):
        acc = 0
        for b in bc.chain:
            if b.header['miner'] == addr:
                acc += 25
            for t in b.body:
                if t['recipient'] == addr:
                    acc += int(t['amount'])
                elif t['sender'] == addr:
                    acc -= int(t['amount'])
        return acc

    class Block:
        def __init__(self, prev_block, transactions, miner):
            self.header = {
                "prev_block": prev_block,
                "version": VERSION,
                "timestamp": time(),
                "miner": miner,
                "nonce": 0,
            }
            self.body = transactions

        @staticmethod
        def create_from_json(header, body):
            b = BlockChain.Block(0, [], 0)
            b.header = header
            b.body = body
            return b


        def get_hash(self):
            return sha256(repr(self.header).encode()).hexdigest()

        def inc_nonce(self):
            self.header['nonce'] += 1

        def get_nonce(self):
            return self.header['nonce']

        def __repr__(self):
            return str({'header': self.header, 'body': self.body})


def main():
    bc = BlockChain()
    while True:
        cmd = input()
        match cmd:
            case 'help':
                print(
                    '''
# note that filename shouldn't contain extension
new - initialize new blockchain
mine - create new block, adds a reward to a given wallet
view - renders all blocks
blnc - shows balance of a given wallet
mkwlt - creates new wallet
wltd - shows values of a wallet by a filename
trns - creates transaction (sender credentials via filename, recipient, amount)
lsops - lists all commited operations
save - saves blockchain to a file with given name
load - loads blockchain from a file via filename
exit - exit the program
                    '''
                )
            case 'new':
                bc = BlockChain()
                print('blockchain initialized')
            case 'mine':
                bc.add_block(input('enter wallet address '))
                print(f'iterations to mine: {bc.chain[-1].get_nonce()}')
            case 'view':
                print(bc.chain)
            case 'blnc':
                print(BlockChain.find_balances(bc, input('enter wallet address: ')))
            case 'mkwlt':
                BlockChain.create_wallet()
            case 'wltd':
                try:
                    print(BlockChain.load_wallet(input('enter wallet credentials: ')))
                except FileNotFoundError:
                    print('not found')
            case 'trns':
                bc.add_transaction(input('enter wallet credentials: '), input("enter receiver's wallet: "), input('amount: '))
            case 'lsops':
                bc.list_operations()
            case 'save':
                bc.save_chain()
            case 'load':
                try:
                    bc.load_chain(input('enter filename: '))
                except FileNotFoundError:
                    print('not found')
                except Exception as e:
                    print(e)
            case 'exit':
                exit()
            case _:
                print('unknown command')


if __name__ == '__main__':
    main()
