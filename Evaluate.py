"""
Evaluate the performance of an open relation extractor.

The original code is written by Suncon Zheng and taken from https://github.com/TJUNLP/NSL4OIE. It is marginally edited by Tom Harting.
Refactoring would be useful, but remains future work.
"""


def evaluation_rel(testresult):
    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    # total_all0 = len(testresult)

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        # print('---')

        # UNCOMMENT TO COMPARE PREDICTION AND TRUE
        # print('ptag--', str(ptag))
        # print('ttag--', str(ttag))

        # if str(ptag) == str(ttag):
        #     total_predict_right += 1.
        #     print('ptag--',str(ptag))
        #     print('ttag--',str(ttag))
        # # else:
        # print('ptag--',str(ptag))
        # print('ttag--',str(ttag))
        # print(str(ptag))
        prel = []
        for p in range(0, len(ptag)):
            if ptag[p].__contains__('R-S'):
                prel.append((p, p + 1))
                # print(ptag[p],'**prel***R-S******', ptag)

            elif ptag[p].__contains__("R-B"):
                j = p + 1
                while j < len(ptag):
                    if ptag[j].__contains__("R-I"):
                        j += 1
                    elif ptag[j].__contains__("R-E"):
                        j += 1
                        prel.append((p, j))
                        # print('**prel*********', ptag)
                        break
                    else:
                        j += 1
                        # break

        trel = []
        for t in range(0, len(ttag)):
            if ttag[t].__contains__("R-S"):
                trel.append((t, t + 1))
                # print('**trel***R-S******', trel)

            elif ttag[t].__contains__("R-B"):
                j = t + 1
                while j < len(ttag):
                    if ttag[j].__contains__("R-I"):
                        j += 1
                    elif ttag[j].__contains__("R-E"):
                        j += 1
                        trel.append((t, j))
                        # print('**trel*********', trel)
                        break
                    else:
                        j += 1
                        # break

        # for i in range(0,len(ttag)):
        # if not ttag[i].__contains__('O'):
        #     total_all0 -=1
        #     break

        flags = 0
        if len(trel) == 1:
            total_right += 1
        if len(prel) == 1:
            total_predict += 1
            if len(prel) == len(trel):
                if str(prel[0]) == str(trel[0]):
                    total_predict_right += 1
                    flags = 1

    print('Length of the test result= ', len(testresult))
    print('Total correct predictions= ', total_predict_right)
    print('Total predictions = ', total_predict)
    print('Total relations present =', total_right)
    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right


def evaluavtion_triple(testresult):
    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        predictrightnum, predictnum, rightnum = count_sentence_triple_num(ptag, ttag)
        total_predict_right += predictrightnum
        total_predict += predictnum
        total_right += rightnum

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F


def count_sentence_triple_num(ptag, ttag):
    # transfer the predicted tag sequence to triple index

    predict_rmpair = tag_to_triple_index(ptag)
    right_rmpair = tag_to_triple_index(ttag)
    predict_right_num = 0  # the right number of predicted triple
    predict_num = 0  # the number of predicted triples
    right_num = 0
    for type in predict_rmpair:
        eelist = predict_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        predict_num += min(len(e1), len(e2))

        if right_rmpair.__contains__(type):
            reelist = right_rmpair[type]
            re1 = reelist[0]
            re2 = reelist[1]

            for i in range(0, min(min(len(e1), len(e2)), min(len(re1), len(re2)))):
                if e1[i][0] == re1[i][0] and e1[i][1] == re1[i][1] \
                        and e2[i][0] == re2[i][0] and e2[i][1] == re2[i][1]:
                    predict_right_num += 1

    for type in right_rmpair:
        eelist = right_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        right_num += min(len(e1), len(e2))
    return predict_right_num, predict_num, right_num


def tag_to_triple_index(ptag):
    rmpair = {}
    for i in range(0, len(ptag)):
        tag = ptag[i]
        if not tag.__eq__("O") and not tag.__eq__(""):
            type_e = tag.split("__")
            if not rmpair.__contains__(type_e[0]):
                eelist = []
                e1 = []
                e2 = []
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist.append(e1)
                eelist.append(e2)
                rmpair[type_e[0]] = eelist
            else:
                eelist = rmpair[type_e[0]]
                e1 = eelist[0]
                e2 = eelist[1]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist[0] = e1
                eelist[1] = e2
                rmpair[type_e[0]] = eelist
    return rmpair
