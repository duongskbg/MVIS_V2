boxStatus = {
    '0000': 'Missing all 4 components', 
    '0001': 'Missing 3 components on top', 
    '0010': 'Missing the left, booklet and cord',
    '0011': 'Missing the left and booklet',
    '0100': 'Missing the left, right and cord',
    '0101': 'Missing the left and right',
    '0110': 'Missing the left and the cord',
    '0111': 'Missing the left',
    '1000': 'Missing the booklet, right and the cord',
    '1001': 'Missing the right and booklet',
    '1010': 'Missing the booklet and cord',
    '1011': 'Missing the booklet',
    '1100': 'Missing the right and the cord',
    '1101': 'Missing the right',
    '1110': 'Missing the cord',
    '1111': 'Box OK' 
    }

class BoxStatistics():
    def __init__(self):
        self.num_occur = 0 # number of occurences
        self.pass_time = 0
        self.fail_time = 0
        self.yield_rate = 0
        self.count_dict = {
            'power_cord_pass' : 0,
            'left_ear_pass' : 0,
            'right_ear_pass': 0,
            'booklet_pass': 0,
            'power_cord_fail': 0,
            'left_ear_fail': 0,
            'right_ear_fail': 0,
            'booklet_fail': 0
            }
        
    def update_stats(self, box_status):
        self.num_occur += 1
        if box_status == '1111':
            self.pass_time += 1
        else:
            self.fail_time += 1
        self.yield_rate = self.pass_time / self.num_occur
        for idx, character in enumerate(box_status):
            if idx == 0:
                if character == '0':
                    self.count_dict['left_ear_fail'] += 1
                else:
                    self.count_dict['left_ear_pass'] += 1                    
            elif idx == 1:
                if character == '0':
                    self.count_dict['booklet_fail'] += 1
                else:
                    self.count_dict['booklet_pass'] += 1 
            elif idx == 2:
                if character == '0':
                    self.count_dict['right_ear_fail'] += 1
                else:
                    self.count_dict['right_ear_pass'] += 1 
            elif idx == 3:
                if character == '0':
                    self.count_dict['power_cord_fail'] += 1
                else:
                    self.count_dict['power_cord_pass'] += 1
                