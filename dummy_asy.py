import asyncio
import math

class Boss:
    def __init__(self, cunts: list['Workers']) -> None:
        self.cunts = cunts
        for cunt in self.cunts:
            cunt.boss = self
        self._interim_vals = []
        self.final_vals = []

    @property
    def prod(self):
        return math.prod(self.interim_vals)
    
    async def interim_vals(self):
        n = 0
        while len(self._interim_vals) < len(self.cunts):
            print(n, len(self._interim_vals))
            n += 1
            await asyncio.sleep(1.0)
        return self._interim_vals

    async def boss_cunts_around(self):
        tasks = [asyncio.create_task(cunt.do_work()) for cunt in self.cunts]
        _ = await asyncio.wait(tasks)
        print(self.final_vals)

 

class Worker:
    def __init__(self, name: str, x: int, y: int, z: int):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.boss = None

    def __repr__(self):
        return f'<Worker({self.name}, {self.x}, {self.y}, {self.z})>'
    
    async def do_work(self):
        a = self.x*self.y
        self.boss._interim_vals.append(a)
        vals = await self.boss.interim_vals()
        b = math.prod(vals)
        c = b + self.z
        self.boss.final_vals.append(c)



async def main():
    cunts = [Worker(f'w{i}', ((i+2)*23)%17, ((i+3)*29)%19, ((i+5)*37)%31) for i in range(4)]
    boss = Boss(cunts)
    await boss.boss_cunts_around()

if __name__ == '__main__':
    print('??')
    asyncio.run(main())