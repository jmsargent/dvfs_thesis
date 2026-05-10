#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mustard.h"

struct TileOperation
{
    std::string             name;
    MatrixTile              write;
    std::vector<MatrixTile> reads;

    std::string toString() const
    {
        auto t = [](int v) { return std::to_string(v); };
        int  r  = write.first, c = write.second;
        if (name == "POTRF" || name == "GETRF")
            return name + "(" + t(r) + ")";
        if (name == "SYRK")
            return name + "(" + t(r) + "," + t(reads[1].second) + ")";
        if (name == "GEMM")
            return name + "(" + t(r) + "," + t(c) + "," + t(reads[1].second) + ")";
        return name + "(" + t(r) + "," + t(c) + ")";
    }

    bool operator<(const TileOperation& other) const
    {
        if (name != other.name) return name < other.name;
        if (write != other.write) return write < other.write;
        return reads < other.reads;
    }

    // Cholesky kernels
    static TileOperation potrf(int k) { return {"POTRF", {k, k}, {{k, k}}}; }
    static TileOperation trsm(int i, int k) { return {"TRSM", {i, k}, {{k, k}, {i, k}}}; }
    static TileOperation syrk(int i, int k) { return {"SYRK", {i, i}, {{i, i}, {i, k}}}; }
    static TileOperation cholGemm(int j, int i, int k)
    {
        return {"GEMM", {j, i}, {{j, i}, {j, k}, {i, k}}};
    }

    // LU kernels
    static TileOperation getrf(int k) { return {"GETRF", {k, k}, {{k, k}}}; }
    static TileOperation trsmL(int k, int i) { return {"TRSM_L", {k, i}, {{k, k}, {k, i}}}; }
    static TileOperation trsmR(int i, int k) { return {"TRSM_R", {i, k}, {{k, k}, {i, k}}}; }
    static TileOperation luGemm(int i, int j, int k)
    {
        return {"GEMM", {i, j}, {{i, j}, {i, k}, {k, j}}};
    }

    enum class Algorithm { Cholesky, LU };

    static TileOperation fromString(const std::string& s, Algorithm alg)
    {
        auto             lparen = s.find('(');
        auto             rparen = s.find(')');
        std::string      op     = s.substr(0, lparen);
        std::vector<int> args;
        if (lparen != std::string::npos && rparen != std::string::npos)
        {
            std::istringstream iss(s.substr(lparen + 1, rparen - lparen - 1));
            std::string        tok;
            while (std::getline(iss, tok, ';'))
                args.push_back(std::stoi(tok));
        }
        return build(op, args, alg);
    }

    static TileOperation build(const std::string& op, const std::vector<int>& args, Algorithm alg)
    {
        if (op == "POTRF")  return potrf(args[0]);
        if (op == "TRSM")   return trsm(args[0], args[1]);
        if (op == "SYRK")   return syrk(args[0], args[2]);
        if (op == "GEMM")   return alg == Algorithm::Cholesky
                                       ? cholGemm(args[0], args[1], args[2])
                                       : luGemm(args[0], args[1], args[2]);
        if (op == "GETRF")  return getrf(args[0]);
        if (op == "TRSM_L") return trsmL(args[0], args[1]);
        if (op == "TRSM_R") return trsmR(args[0], args[1]);
        throw std::runtime_error("Unknown op: " + op);
    }
};
