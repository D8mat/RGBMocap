"""Standalone protocol/math helpers. No Max calls on the network thread."""
import hashlib,json,math,queue,socket,struct,threading
LIMIT=1024*1024

def transpose(a):return [list(x) for x in zip(*a)]
def multiply(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def rotation(m):return [r[:3] for r in m]
def position(m):return [r[3] for r in m]
def matrix(values):
    if len(values)!=12 or not all(math.isfinite(x) for x in values):raise ValueError('Invalid bone matrix')
    return [list(values[i:i+4]) for i in (0,4,8)]
def validate(schema):
    bones=schema['bones'];n=len(bones)
    if schema.get('version')!=2 or not 1<=n<=256:raise ValueError('Unsupported skeleton')
    if len(schema['actor'])!=64:raise ValueError('Invalid actor ID')
    bytes.fromhex(schema['actor'])
    if not 1<=schema['slot']<=128:raise ValueError('Invalid actor slot')
    names=set()
    for i,b in enumerate(bones):
        if len(b)!=4 or not isinstance(b[0],str) or not b[0] or b[0] in names or not -1<=b[1]<i:raise ValueError('Invalid hierarchy')
        names.add(b[0]);matrix(b[2])
    return hashlib.sha256(json.dumps(bones,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def snapshot(raw,slots):
    if len(raw)<10:raise ValueError('Truncated snapshot')
    seq,count=struct.unpack_from('<QH',raw);offset=10;poses=[]
    if count>128:raise ValueError('Too many actors')
    for _ in range(count):
        slot,frame=struct.unpack_from('<II',raw,offset);offset+=8
        schema=slots[slot];n=len(schema['bones']);size=n*48
        vals=struct.unpack_from('<'+'f'*(n*12),raw,offset);offset+=size
        poses.append((schema['actor'],frame,[matrix(vals[i:i+12]) for i in range(0,len(vals),12)]))
    # Optional selected-person metadata follows a Bake request. No facial negotiation.
    if offset<len(raw):
        selected,=struct.unpack_from('<H',raw,offset);offset+=2+selected*32
    if offset!=len(raw):raise ValueError('Invalid snapshot length')
    return poses

class Client:
    def __init__(self,port):
        self.events=queue.Queue(64);self.out=queue.Queue(16);self.stop=threading.Event();self.sock=None
        self.thread=threading.Thread(target=self.run,args=(port,),daemon=True);self.thread.start()
    def emit(self,tag,data):
        while not self.stop.is_set():
            try:self.events.put((tag,data),timeout=.1);return
            except queue.Full:pass
    def send(self,data):self.out.put_nowait(data)
    def close(self):
        self.stop.set()
        if self.sock:
            try:self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            self.sock.close()
    def run(self,port):
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=2) as sock:
                self.sock=sock;sock.settimeout(.05);buf=bytearray();self.emit('connected',b'')
                while not self.stop.is_set():
                    try:
                        while True:sock.sendall(self.out.get_nowait())
                    except queue.Empty:pass
                    try:data=sock.recv(65536)
                    except socket.timeout:continue
                    if not data:raise ConnectionError('RGB MoCap disconnected')
                    buf.extend(data)
                    while len(buf)>=4:
                        size,=struct.unpack_from('<I',buf)
                        if not 1<=size<=LIMIT:raise ValueError('Invalid packet size')
                        if len(buf)<size+4:break
                        self.emit(chr(buf[4]),bytes(buf[5:4+size]));del buf[:4+size]
        except Exception as exc:
            if not self.stop.is_set():self.emit('error',str(exc))
        finally:self.sock=None
