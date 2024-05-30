# 从pcap中读取时间间隔内的数据包
from scapy.all import *

def process_packets(packets,dict_addr):
    original_sequence = []
    timeStamps = []

    for packet in packets:
        src_ip = packet.src
        dst_ip = packet.dst
        pkt_len = len(packet)
        pkt_time = packet.time
        current_pair = (src_ip, dst_ip)
        
        if current_pair in dict_addr:
            pkt_len *= 1
        else:
            pkt_len *= -1
        # Add the last interval
        original_sequence.append(pkt_len)
        timeStamps.append(pkt_time)
    
    D0_sequence = [x if x > 0 else 0 for x in original_sequence]
    U0_sequence = [x if x < 0 else 0 for x in original_sequence]

    D0_Cumulative_Sum_sequence = []
    U0_Cumulative_Sum_sequence = []

    cumulative_sum = 0
    # 遍历原始列表
    for num in D0_sequence:
        # 累加当前元素到累计和
        cumulative_sum += num
        # 将累计和添加到累计和列表中
        D0_Cumulative_Sum_sequence.append(cumulative_sum)

    cumulative_sum = 0
    for num in U0_sequence:
        # 累加当前元素到累计和
        cumulative_sum += num
        # 将累计和添加到累计和列表中
        U0_Cumulative_Sum_sequence.append(cumulative_sum)
    
    combined_list = zip(U0_sequence,D0_sequence,U0_Cumulative_Sum_sequence,D0_Cumulative_Sum_sequence,timeStamps)

    return combined_list

def write_results_to_file(results, output_file):
    with open(output_file, 'w') as f:
        f.write("U0_sequence,D0_sequence,U0_Cumulative_Sum_sequence,D0_Cumulative_Sum_sequence,timeStamps\n")
        for result in results:
            f.write(','.join(map(str, result)) + '\n')



if __name__ == "__main__":

    for root, dirs, files in os.walk("/home/lxc/Datasets/ISCX-VPN-NonVPN/ISCX-NonVPN"):
        for file in files:
            print("cur file:",file)
            pcap_file = os.path.join(root, file)
            output_file = "/home/lxc/ETC-PS/data/nonvpn/"+pcap_file.split("/")[-1].replace(".pcap","")
            if os.path.exists(output_file):
                print("processed:",output_file)
                continue
            packets = rdpcap(pcap_file)

            # 确定每个数据包的方向
            dict_addr = []
            dict_addr_reverse = []
            for packet in packets:
                 # 获取源地址和目标地址
                src_ip = packet.src
                dst_ip = packet.dst
                # 初始化一个空的列表来存储地址对
                current_pair = (src_ip, dst_ip)
                reverse_pair = (dst_ip, src_ip)
                # print(current_pair)
                # 全新的
                if current_pair not in dict_addr and reverse_pair not in dict_addr_reverse and \
                    reverse_pair not in dict_addr:
                    dict_addr.append(current_pair)
                    dict_addr_reverse.append(reverse_pair)
            
            print(dict_addr)
            packets.sort(key=lambda x: x.time)
            results = process_packets(packets,dict_addr)
            write_results_to_file(results, output_file)