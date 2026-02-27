# L-Representation

L-Representation (L-Rep) is a method for representing any discrete or continuous mathematical-geometric object (point, vector, curve, surface, manifold, function, spectral coefficients, SDF, etc.) as a single integer **L**. The goal is to enable geometric/numerical manipulation of the entire object using integer operations (add/mul/shift/bitwise) with controlled precision, and to implement it efficiently on dedicated hardware (FPGA/ASIC). This document presents: (1) the precise mathematical model of the encoding, (2) proof of the conditions for integer operations to be equivalent to the desired geometric operation, (3) algorithms/strategies for overflow and precision resolution, (4) feasible hardware design (L-ALU, ISA, pipeline), and (5) implementation roadmap.

L-Representation (L-Rep) là một phương pháp biểu diễn mọi đối tượng toán-hình học rời rạc hoặc liên tục (điểm, vector, đường cong, bề mặt, đa tạp, hàm, spectral coefficients, SDF, v.v.) thành một số nguyên duy nhất **L**. Mục tiêu: cho phép thao tác hình học/phép toán số trên toàn bộ đối tượng bằng các phép toán nguyên (add/mul/shift/bitwise) với độ chính xác được kiểm soát, và thực hiện hiệu quả trên phần cứng chuyên dụng (FPGA/ASIC). Tài liệu này đưa ra: (1) mô hình toán học chính xác của mã hóa, (2) chứng minh điều kiện để các phép toán nguyên tương đương với phép toán hình học mong muốn, (3) thuật toán/chiến lược giải quyết overflow và precision, (4) thiết kế phần cứng khả thi (L-ALU, ISA, pipeline), và (5) lộ trình triển khai.

This work is an early-stage theoretical exploration developed independently by a student researcher. Due to practical constraints, the current version focuses on conceptual formulation and preliminary validation. The author welcomes feedback, critique, and collaboration from the community.


# Access : 

**Major languages:** English, Vietnamese

**Recommended reading:**

**Experiment :** https://github.com/nahhididwin/L-Representation/tree/main/experiment


# Author :

Full name : Dang Dinh Phu Hung (Đặng Đình Phú Hưng)

Nation : Vietnam

City : Ho Chi Minh City

Date of birth: https://github.com/nahhididwin/L-Representation/blob/main/main/dob.txt


Github : https://github.com/nahhididwin


# License :

Read : https://github.com/nahhididwin/L-Representation?tab=License-1-ov-file

Repositories Public Date (https://github.com/nahhididwin/L-Representation) : (DD/MM/YYYY)
