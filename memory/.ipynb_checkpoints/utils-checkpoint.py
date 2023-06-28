import random
class SumTree:
    def __init__(self, buffer_size):
        self.size = 0
        self.scores = [0]*(4*buffer_size + 1)
        self.elements = [0]*buffer_size
        self.buffer_size = buffer_size
        self.ep = 1e-6
        self.max_priority = 0
    
    def propogate(self,l,r,score_index,idx,add_value):
        self.scores[score_index] += add_value
        if r > l:
            mid = int((l+r)/2)
            if idx > mid:
                self.propogate(mid+1,r,2*score_index+2,idx,add_value)
            else:
                self.propogate(l,mid,2*score_index+1,idx,add_value)
    
    def insert(self,element):
        """
            Inserting from the start if the buffer is full
            which means making sure that newest elements are retained.
        """
        self.max_priority =  max([self.max_priority,element.score])
        element._replace(score=self.max_priority)
        idx = self.size % self.buffer_size
        add_value = element.score - (0 if (self.size < self.buffer_size) else self.elements[idx].score)
        self.elements[idx] = element        
        self.size += 1
        self.propogate(0,self.buffer_size - 1,0,idx,add_value)
        
    def update(self,idx, score):
        add_value = score - self.elements[idx].score
        self.elements[idx]._replace(score=score)
        if score > self.max_priority:
            self.max_priority = score
        self.propogate(0,self.buffer_size - 1,0,idx,add_value)
    
    def random_retrieve(self):
        def recurse(l,r,score_index):
            if r > l:
                mid = int((l+r)/2)
                scr_l = float(self.scores[2*score_index+1])/self.scores[score_index]
                rand = random.random()
                rand = (1. - 2*self.ep)*rand + self.ep
                if rand < scr_l:
                    return recurse(l,mid,2*score_index + 1)
                else:
                    return recurse(mid+1,r,2*score_index + 2)
            else:
                return l,self.elements[l]
        return recurse(0,self.buffer_size-1,0)
    
    def sample(self,n):
        samples = [self.random_retrieve() for i in range(n)]
        ids = [samples[i][0] for i in range(n)] 
        experiences = [samples[i][1] for i in range(n)]
        return ids,experiences
    
    def length(self):
        """Return the current size of internal memory."""
        return min([self.size, self.buffer_size])