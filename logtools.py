import re
# from icecream import ic
from typing import TypeAlias, Hashable
from utils import collect
from datetime import datetime as dt
from m import MTuple, MCounter

Response: TypeAlias = tuple[str|int|None, int|None]



class MiniLog(list):
    class PassThrough:
        def __init__(self, minilog):
            self.minilog = minilog

        def __call__(self, *args):
            self.minilog(*args)
            return MTuple(*args) if len(args)!=1 else args[0]

    def __init__(self, *firstline):
        super().__init__(firstline)
        self.counter = MCounter()
        self.meta = {}

    def __call__(self, *content):
        self.append(content)

    def count(self, thing: Hashable, d: int=1):
        self.counter[thing] += d
        self.append((thing, d))

    def zero(self, thing: Hashable):
        self.counter[thing] = 0
        self.append((f'Setting count of {thing} to 0'))

    def flush(self):
        flush = list(self)
        self.clear
        return flush
    
    # L = ml.pass_through()
    # args = (2,3)
    # out = L('logme1', sum(*args), 'sum', *args, 'logme2')[1]
    # this will log ('logme1', 5, 'sum', 2, 3, 'logme2') and `out` will be 5
    # out = L('logme1', sum(*args), 'sum', args[0]*args[1], 'product', *args, 'logme2')[1, 3]
    # this will log ('logme1', 5, 'sum', 6, 'product', 2, 3, 'logme2') and `out` will be 5, 6
    def pass_through(self):
        return self.PassThrough(self)
    
    def show(self):
        MLDialogue(self)()
    
    def write(self, clear=True) -> str:
        n = dt.now()
        fname = f'minilog_{n.year}_{n.month}_{n.day}_{n.hour}_{n.minute}_{n.second}_{n.microsecond}.log'
        with open(fname, 'wt') as f:
            for i, line in self.log:
                f.write(f'>> {i} ::')
                for item in line:
                    f'\t{item}'
        if clear:
            self.clear()
        return fname

    
class MLDialogue:

    OPTS = 'View all? <Y.*>\nView some? <a: int, b: int>\nView one? <a: int>\nView none? <anything>\n:: '

    def __init__(self, ml: MiniLog) -> None:
        self.ml = ml

    def supersplit(self, input: list[str]|str, *seps, ws_default=True, ws_always=False) -> list[str]:
        input = collect(input, list, empty_if_none=True)
        if (not seps and ws_default) or (ws_always and '' not in seps):
            seps = ('',) + seps
        outlist = []
        for item in input: 
            outlist += item.split(seps[0]) if seps[0] else item.split()
        outlist = [i for i in outlist if i]
        if len(seps)==1:
            return outlist
        return self.supersplit(outlist, *seps[1:], ws_default=False)
    
    def confirm(self, instr: str, interp: str, tbc: Response) -> Response:
        confirm_ = input(
            f"You entered {instr}. Did you mean {interp}?\n" +
            "Enter 'y' for yes, 'n' for no, or anything else to quit.\n" +
            ":: "
        )
        y, n = confirm_.lower()=='y', confirm_.lower()=='n'
        if n:
            return self.parse_input(
                input(
                    'Try again.\n' +
                    MLDialogue.OPTS
                )
            )
        elif not y:
            return None, None
        else:
            return tbc
        
    def parse_input(self, instr: str):
        if not instr:
            return None, None
        instr_l = self.supersplit(instr, '', *list("\"',:;-"))
        if instr_l[0].lower().startswith('y'):
            resp = ('y', None)
            if instr_l[0].lower() not in ['y', 'yes']:
                return self.confirm(instr, 'yes', resp)
            return resp
        if re.fullmatch(f'[0-9][0-9]*', instr_l[0]):
            start, end = int(instr_l[0]), None
            hmm = False
            if len(instr_l) == 1:
                return start, None
            if re.fullmatch(f'[0-9][0-9]*', instr_l[1]):
                end = int(instr_l[1])
                if end < start:
                    start, end = end, start
                end = end if end-start else None
                if len(instr_l) > 2:
                    hmm = True
            else:
                hmm = True
            if hmm:
                return self.confirm(instr, (start, end) if end else start, (start, end))
            return start, end
        return None, None

    def __call__(self, _again=False):
        continyoo = True
        pl = len(self.log)!=1
        while continyoo:
            request = (
                "Again?" 
                if _again 
                else (
                    f'There {"are" if pl else "is"} {len(self.log)} '+
                    f'log entr{"ies" if pl else "y"}.\n'
                )
            )
            again = True
            i0, i1 = self.parse_input(input(request + MLDialogue.OPTS))
            if i0 is None:
                continyoo = False
            elif i0 == 'y':
                print(self.log)
            elif i1 is None:
                print(self.log[i0])
            else:
                print(self.log[i0:i1])
