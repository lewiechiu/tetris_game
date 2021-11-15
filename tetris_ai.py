#!/usr/bin/python3
# -*- coding: utf-8 -*-

from tetris_model import BOARD_DATA, Shape
import math
from time import perf_counter
from datetime import datetime
import numpy as np


class TetrisAI(object):

    def nextMove(self):
        t1 = perf_counter()
        if BOARD_DATA.currentShape == Shape.shapeNone:
            return None

        currentDirection = BOARD_DATA.currentDirection
        currentY = BOARD_DATA.currentY
        _, _, minY, _ = BOARD_DATA.nextShape.getBoundingOffsets(0)
        nextY = -minY

        # print("=======")
        strategy = None
        if BOARD_DATA.currentShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            direction_available = (0, 1)
        elif BOARD_DATA.currentShape.shape == Shape.shapeO:
            direction_available = (0,)
        else:
            direction_available = (0, 1, 2, 3)

        if BOARD_DATA.nextShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            next_shape_direction_available = (0, 1)
        elif BOARD_DATA.nextShape.shape == Shape.shapeO:
            next_shape_direction_available = (0,)
        else:
            next_shape_direction_available = (0, 1, 2, 3)

        best_score = -1000
        next_position = []
        for direction in direction_available:
            shape_left_x, shape_right_x, _, _ = BOARD_DATA.currentShape.getBoundingOffsets(direction)
            for x0 in range(-shape_left_x, BOARD_DATA.width - shape_right_x):
                board = self.calcStep1Board(direction, x0)
                for next_shape_direction in next_shape_direction_available:
                    next_shape_left_x, next_shape_right_x, _, _ = BOARD_DATA.nextShape.getBoundingOffsets(next_shape_direction)

                    dropDist = self.calc_next_drop_dist(
                        board,
                        next_shape_direction,
                        range(-next_shape_left_x, BOARD_DATA.width - next_shape_right_x)
                    )

                    for x1 in range(-next_shape_left_x, BOARD_DATA.width - next_shape_right_x):
                        current_score = self.calculate_score(np.copy(board), next_shape_direction, x1, dropDist)
                        if current_score > best_score:
                            best_score = current_score
                            next_position = [direction, x0]

        print(f"[STATUS] {(perf_counter() - t1):.3f} sec, score: {best_score:.3f}")
        return next_position + [best_score]

    def calc_next_drop_dist(self, data, direction, xRange):
        res = {}
        for x0 in xRange:
            if x0 not in res:
                res[x0] = BOARD_DATA.height - 1
            for x, y in BOARD_DATA.nextShape.getCoords(direction, x0, 0):
                yy = 0
                while yy + y < BOARD_DATA.height and (yy + y < 0 or data[(y + yy), x] == Shape.shapeNone):
                    yy += 1
                yy -= 1
                if yy < res[x0]:
                    res[x0] = yy
        return res

    def calcStep1Board(self, direction, x0):
        board = np.array(BOARD_DATA.getData()).reshape((BOARD_DATA.height, BOARD_DATA.width))
        self.dropDown(board, BOARD_DATA.currentShape, direction, x0)
        return board

    def dropDown(self, data, shape, direction, x0):
        dy = BOARD_DATA.height - 1
        for x, y in shape.getCoords(direction, x0, 0):
            yy = 0
            while yy + y < BOARD_DATA.height and (yy + y < 0 or data[(y + yy), x] == Shape.shapeNone):
                yy += 1
            yy -= 1
            if yy < dy:
                dy = yy
        # print("dropDown: shape {0}, direction {1}, x0 {2}, dy {3}".format(shape.shape, direction, x0, dy))
        self.dropDownByDist(data, shape, direction, x0, dy)

    def dropDownByDist(self, data, shape, direction, x0, dist):
        for x, y in shape.getCoords(direction, x0, 0):
            data[y + dist, x] = shape.shape

    def calculate_score(self, step1Board, next_shape_direction, x1, dropDist):
        # print("calculate_score")
        t1 = datetime.now()
        width = BOARD_DATA.width
        height = BOARD_DATA.height

        self.dropDownByDist(step1Board, BOARD_DATA.nextShape, next_shape_direction, x1, dropDist[x1])
        # print(datetime.now() - t1)

        # Term 1: lines to be removed
        fullLines, nearFullLines = 0, 0
        roofY = [0] * width
        holeCandidates = [0] * width
        holeConfirm = [0] * width
        vHoles, vBlocks = 0, 0
        for y in range(height - 1, -1, -1):
            hasHole = False
            hasBlock = False
            for x in range(width):
                if step1Board[y, x] == Shape.shapeNone:
                    hasHole = True
                    holeCandidates[x] += 1
                else:
                    hasBlock = True
                    roofY[x] = height - y
                    if holeCandidates[x] > 0:
                        holeConfirm[x] += holeCandidates[x]
                        holeCandidates[x] = 0
                    if holeConfirm[x] > 0:
                        vBlocks += 1
            if not hasBlock:
                break
            if not hasHole and hasBlock:
                fullLines += 1
        vHoles = sum([x ** .7 for x in holeConfirm])
        maxHeight = max(roofY) - fullLines
        # print(datetime.now() - t1)

        roofDy = [roofY[i] - roofY[i+1] for i in range(len(roofY) - 1)]

        if len(roofY) <= 0:
            stdY = 0
        else:
            stdY = math.sqrt(sum([y ** 2 for y in roofY]) / len(roofY) - (sum(roofY) / len(roofY)) ** 2)
        if len(roofDy) <= 0:
            stdDY = 0
        else:
            stdDY = math.sqrt(sum([y ** 2 for y in roofDy]) / len(roofDy) - (sum(roofDy) / len(roofDy)) ** 2)

        absDy = sum([abs(x) for x in roofDy])
        maxDy = max(roofY) - min(roofY)
        # print(datetime.now() - t1)

        score = fullLines * 1.8 - vHoles * 1.0 - vBlocks * 0.5 - maxHeight ** 1.5 * 0.02 \
            - stdY * 0.0 - stdDY * 0.01 - absDy * 0.2 - maxDy * 0.3
        # print(score, fullLines, vHoles, vBlocks, maxHeight, stdY, stdDY, absDy, roofY, direction, x0, next_shape_direction, x1)
        return score


TETRIS_AI = TetrisAI()

