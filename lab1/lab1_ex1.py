class Competitor:
    def __init__(self, name, surname, country, score):
        self.name = name
        self.surname = surname
        self.country = country
        self.score = score  # will just store the useful score

comp_list = list()
country_score = dict()

with open("/home/oormacheah/Desktop/Uni shit/MLPR/lab1/score.txt", "r") as f:
    for line in f:
        line = line.rstrip()
        # format the split for taking some whitespaces
        elements = line.split(' ', 3)   
        # 4th element is the list with the scores
        tmp_scores = list(map(float, elements[3].split()))
        del elements[3]
        tmp_scores.remove(max(tmp_scores))
        tmp_scores.remove(min(tmp_scores))
        
        elements.append(sum(tmp_scores)) # added the float sum of relevant scores

        if elements[2] in country_score:
            country_score[elements[2]] += elements[3]
        else:
            country_score[elements[2]] = elements[3]
        
        comp_list.append(Competitor(*elements))

comp_list.sort(key=lambda e: e.score, reverse=True)
del comp_list[3:]

print('Best competitors are: ')

for index, value in enumerate(comp_list):
    print(str(index + 1) + '. ' + value.name + ' ' + value.surname + ' -- ' + str(value.score))

max_score = 0.0

for cauntri in country_score:
    if country_score[cauntri] > max_score:
        max_score = country_score[cauntri]
        best_country = cauntri

print('Best country:')
print('%s -- score: %.2f' % (best_country, country_score[best_country]))